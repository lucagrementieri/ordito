# Role: NVIDIA Warp Expert

You are an expert in NVIDIA Warp (`wp`) working on **triwarp**, a GPU geometry-processing library.
Follow every rule below when writing kernels, `@wp.func` helpers, Python-scope wrappers, tests and
benchmarks.

**This file has two parts, and they answer different questions.**

- **Part I — Rules (§1-§11).** How code in this repository must be written. Obey these.
- **Part II — Measured facts (§12-§16).** What has already been measured on this hardware and this
  Warp version: platform bugs, a cost model, kernel-shape verdicts, measurement traps, and the
  status of specific triwarp functions. Read the relevant section **before** proposing an
  optimization or diagnosing a slowdown — most plausible ideas in this repository have already been
  built and priced, and several of them lost.

**Part II is the knowledge base, and it is the only one.** There is no separate memory store: a new
measured finding — a ratio, a refuted idea, a platform quirk, a reference library's trap — is written
into the section of Part II that owns it, next to the numbers it revises or contradicts, and a rule
that follows from it goes into Part I with a cross-reference. Do not start a parallel notes file; a
finding recorded anywhere else is a finding the next reader will re-derive.

Two conventions that hold throughout:

- **A measured decline is a result.** When a change is measured and rejected, the number is written
  at the site *and* recorded here. Do not re-derive it; do not re-propose it without new evidence.
- **A Warp version claim is spelled `Warp 1.17`** — the word immediately before the number. The
  staleness check (§4.5, check 9) reads only that anchored form, because this package writes
  measured ratios in the same shape (`within 1.25x`) and a bare `1.N` token matches an order of
  magnitude more of those than real claims. Installed today: **warp-lang 1.17.0** on an
  **RTX 5090**.

## Contents

**Part I — Rules**

1. [Kernel syntax and semantics](#1-kernel-syntax-and-semantics) — decorators, types, casts,
   kernel-scope restrictions, arithmetic spellings, mutable accumulators
2. [Kernel architecture and launches](#2-kernel-architecture-and-launches) — launch discipline,
   `wp.block_dim()`, block-per-item, fusing, `wp.overload`, `wp.ref`, factories, `@wp.struct`,
   per-thread storage
3. [Python-scope wrappers](#3-python-scope-wrappers) — layout, `triwarp.typing`, allocation, gather,
   `wp.map`, dtype conversion, `BsrMatrix`, the NumPy policy, device rules, readbacks
4. [Evolving the public API](#4-evolving-the-public-api) — naming, signatures, docstring agreement,
   moving/renaming, the 27 mechanical checks
5. [Function ordering within a module](#5-function-ordering-within-a-module)
6. [Documentation](#6-documentation-zensical--mkdocstrings)
7. [Testing](#7-testing) — conventions, devices, fixtures, the parity gate, shared helpers, the nine
   reference libraries
8. [Tooling and local validation](#8-tooling-and-local-validation)
9. [Performance work: measure before you change](#9-performance-work-measure-before-you-change)
10. [Warp API reference mirrors](#10-warp-api-reference-mirrors)
11. [Running long commands](#11-running-long-commands-never-poll-with-until)

**Part II — Measured facts**

12. [The Warp platform](#12-the-warp-platform-bugs-quirks-and-version-status) — memory safety, CPU
    tiles, `wp.ref`, numerics, casts, compilation, `warp.sparse`, builtin verdicts, `wp.Volume`,
    upgrade discipline
13. [The cost model (RTX 5090)](#13-the-cost-model-rtx-5090) — host per-call, device/memory, tuning
    constants
14. [Kernel-shape verdicts](#14-kernel-shape-verdicts) — what wins, what is refuted, when a
    producer-consumer fusion pays, and when a launch buys less than a block barrier
15. [Benchmark and measurement traps](#15-benchmark-and-measurement-traps)
16. [triwarp component status](#16-triwarp-component-status) — open defects, refuted plans, and
    the rules the optimization rounds left behind

---
---

# PART I — RULES

---

## 1. Kernel syntax and semantics

### 1.1 Decorators

- `@wp.kernel` for entry points launched via `wp.launch()`; `@wp.func` for helpers called from
  kernels or other `@wp.func`s.
- **`@wp.kernel` functions MUST NOT return a value.** Annotate `-> None` and write into output
  arrays.
- **Every argument of both `@wp.kernel` and `@wp.func` MUST be explicitly typed.**
- `wp.tid()` may be called **only inside `@wp.kernel`**, never inside `@wp.func` — pass the thread
  index as an argument.
- `@wp.func` may return several values as a tuple; declare `tuple[T1, T2, ...]`. **The generic
  forms parse on Warp 1.17** — `tuple[Any, Any, Any]`, `tuple[wp.Float, wp.Float]` and the mixed
  `tuple[Any, wp.Float]` all compile on an `Any`-generic helper, so a dtype- or rank-generic helper
  is no reason to leave the annotation off.
- A `@wp.func` may read `values.shape[1]` and may carry **default argument values**. `wp.launch`
  also accepts **`None` for an array argument** (a null descriptor) — legal as long as no thread
  indexes it.

### 1.2 Type standards

- Use subscript-style array annotations: `wp.array[wp.vec3]`, `wp.array[wp.int32]`,
  `wp.array[wp.float32]`. Check 14 scans the **whole package** for this, because only in an
  *annotation* position is `wp.array(dtype=T)` the stale spelling rather than a legal allocation.
- In **`triwarp/kernels/` only**: `wp.array2d[T]`, `wp.array3d[T]`, `wp.array4d[T]` for
  multi-dimensional kernel arguments, with matching multi-index `wp.tid()` unpacking. In Python
  wrappers use the `triwarp.typing` aliases instead (§3.2).
- Built-in vector/matrix types: `wp.vec2/3/4`, `wp.mat22/33/44`, `wp.quat`.
  `wp.indexedarray[wp.vec3]` (and its siblings) for sparse/indexed access patterns.
- **No bare `bool` / `int` / `float` annotation** in a `@wp.kernel` / `@wp.func` signature
  (check 18). This reads kernel-scope signatures only — a kernel **factory** is ordinary Python and
  its `row_size: int` / `name: str` parameters are correct, which is why `str` is not in the check's
  table.
- **`wp.constant()` is not what makes a module-level value visible in kernel scope, and wrapping
  with it is optional.** On Warp 1.17 it is `return x` after an `is_value(x)` check (§12.6) — any
  module-level global that evaluates to a scalar/vector/matrix is resolvable from kernel scope
  whether or not it passed through `wp.constant()`. What *is* load-bearing is the typed
  constructor:
  ```python
  TOLERANCE_MERGE = wp.float32(1e-8)
  ```
  A plain Python float works too but is treated as `wp.float32` in kernel-scope arithmetic, and
  mixing that default against a `float64` kernel variable is a parse error — use the
  `wp.float64(...)` constructor where 64-bit precision is required. `triwarp/constants.py` carries
  no `wp.constant()` calls for this reason; do not add one on the theory that it is required.
- **Prefer dtype-generic `@wp.func`s** — `wp.Float` / `wp.Scalar` for scalars, `Any` for vectors,
  the `kernels/predicates.py` convention — as long as the dispatch stays readable. Coverage and its
  limits: §12.4.
- **The generic axis is not always the dtype: `Any` is generic over the *rank* and the *dimension*
  too.** A helper pinned to either breeds copies exactly as a precision-pinned one does. Two merged
  instances: a 5x5 and a 6x6 Householder normal-equation solve differed only in a `for k in
  range(N)` singularity test — because **a matrix has no readable `.shape` in kernel scope**
  (`r.shape[0]` is a parse-time `WarpCodegenAttributeError`) — and the rank-free spelling is a
  reduction over the diagonal, `wp.min(wp.abs(wp.get_diag(r))) < tol`, now
  `linalg.solve_normal_equations`; and a Cramer's-rule barycentric solve lived once per *dimension*
  (`wp.vec2`, `wp.vec3`) although `wp.length_sq` and `wp.dot` say nothing about the ambient
  dimension. **When two bodies differ only in a size, look for the one statement that names it and
  ask whether Warp has a reduction for it.**

### 1.3 Casts and conversions

- Cast `wp.tid()` explicitly when used as an array index: `f = wp.int32(wp.tid())`. **`wp.int32` /
  `wp.float32` are the tree's only cast spelling** — never the bare `int(...)` / `float(...)`
  builtins (check 16 rejects them inside a kernel or `@wp.func` body). They are the same builtins
  under a different name (Warp writes an unconditional `#define int(x) cast_int(x)` into every
  module header), with one difference that matters: **`float(...)` is a hard compile error inside a
  `wp.Float`-generic function** — `total / float(count)` fails to parse with `Input types must be
  the same, got ['float64', 'float32']` rather than silently narrowing. Where the enclosing function
  is or could be generic, the spelling is `type(x)(...)` (the `kernels/predicates.py` convention).
  Detail in §12.5.
- **A cast to the type a value already has is noise; delete it.** No cast is load-bearing: a bare
  `wp.tid()` passes unchanged to a `wp.int32` parameter, to a `wp.Scalar` generic and into a
  kernel-scope slice. So `wp.int32(offsets[i])` on a `wp.array[wp.int32]`, `wp.int32(n_faces)` on a
  `wp.int32` argument, and `wp.int32(f)` three lines under `f = wp.int32(wp.tid())` all say nothing.
- **The tid cast is the one exception the tree keeps**, because it is the *declarative* one: it
  names the type of the index the whole kernel is written against. **Check 22** enforces it — no
  bare single-index `wp.tid()`. It reads single-`Name` assignment targets only: a multi-index
  `i, j = wp.tid()` cannot carry a cast, and a scan keying on the *call* instead would report
  dozens of correct sites as defects (`test_bare_tid_scan_ignores_multi_index_unpacks` pins that).
- A cast of a bare **literal** is never redundant — `wp.int32(0)` is a mutable Warp dynamic variable
  where `0` is a compile-time constant that freezes the enclosing loop (§1.6).
- **A constructor of the type a value already has is the same noise, and the cast scan does not see
  it.** `wp.vec3(*(vertices[face[1]] - vertices[face[0]]))` splats a `wp.vec3` and rebuilds it; the
  difference of two `wp.vec3`s is already a `wp.vec3`. Check 16 classifies *casts*, so a
  `wp.vecN(*(...))` / `wp.matNM(*(...))` splat-and-reconstruct survives it. When deleting one class
  of no-op conversion, grep the constructor spelling too, and read the operand's type rather than
  trusting the scan's silence. (Python-scope constructions from host sequences are not this defect.)
- `wp.cast(expr, TargetType)` is an explicit conversion **only between types of the same size** — it
  is a bit reinterpretation, and a width change fails at *NVRTC* time with
  `static assertion failed with "source and destination must have the same size"`, not at Python
  time. `wp.cast(cross(a, b), wp.vec3)` is fine; `wp.cast(x_f64, wp.float32)` does not compile.
  Widening or narrowing a scalar is the **constructor**: `wp.float32(x)`, `wp.float64(i)`.
  Whole-array conversion at Python scope is a different API — §3.6.

### 1.4 Kernel-scope restrictions

Not supported inside `@wp.kernel` / `@wp.func`:

- Lambdas, list comprehensions, sets, dicts, `list.append()`, `eval()`, recursion, exceptions.
- Python tuples for initialization — use typed constructors: `wp.vec3(1.0, 2.0, 3.0)`.
- For small fixed-size collections use vector types; for larger ones `wp.zeros(shape=N, dtype=T)`
  (stack-allocated) — but see §2.9, which measures that as a **2x loss** against
  `wp.types.vector(length=N)`.
- **A tuple cannot be subscripted by a runtime index; a vector can.** The `tuple[T, T, T]` a
  `@wp.func` returns unpacks into names and nothing more, so `corner[k]` for a loop variable `k`
  does not compile — which is why a handful of kernels reach for the flat slice
  `faces[f * 3 : (f + 1) * 3]`. The spelling that keeps both is `wp.vec3i(*corner_triple(faces, f))`,
  which accepts a runtime `[]` and takes no sub-array view. The rule is **not** "no slices" but *no
  slice where an index form exists*.
- **Variable scope inside conditionals differs from CPython**: a variable defined only inside an
  `if` branch is accessible afterwards in Warp but uninitialized if the branch was not taken —
  always initialize before branching.
- `wp.asin()` / `wp.acos()` auto-clamp to [-1, 1]; an explicit `wp.clamp` before them is redundant
  but harmless.

### 1.5 Arithmetic and conditional spellings

- **`%` follows C++11 semantics** (sign of result = sign of dividend), not Python's.
- **`//` truncates toward zero like `/`, and on integers the two operators are the same
  operation.** `[-8, -7, -1, 0, 1, 7, 8] ÷ 3` gives `[-2, -2, 0, 0, 0, 2, 2]` for both spellings,
  where CPython's `//` gives `[-3, -3, -1, 0, 0, 2, 2]`. Consistent with the `%` rule.
- **Spell integer division `//`** (check 17). `/` on two `int32`s reads as real division and only
  truncates because the operands happen to be integers, so a reader has to recover the types before
  knowing what the line does. Every dividend in this package is a non-negative index, where the two
  conventions coincide; the hazard is *porting* a line with a negative dividend between host Python
  and kernel scope, which changes its answer silently. The scan types an operand only **by
  declaration** — an annotated `wp.int*` / `wp.uint*` / `wp.Int` parameter, an element of an array
  whose annotated dtype is one of those, a module-level integer `wp.constant`, an integer literal, a
  `.shape[...]`, `wp.tid()`, an integer constructor, or an integer-preserving expression over those
  — because a scan that misfires on float division gets switched off by the first person it annoys.
- **Kernel-scope `and` / `or` short-circuit (Warp 1.17).** `codegen.emit_BoolOp` guards each
  later operand behind an `if` on the result so far, so `i + 1 < n and a[i + 1] == v` never reads
  past the end. A comment or a nested-`if` shape justified by "`and` does not short-circuit" is
  stale; two such comments were corrected in the 2026-09-26 pass (§16.25).
- **Spell the remainder `%`, not `i - (i // stride) * stride`.** The long form is the same operation
  but reads as though it is *avoiding* `%` for a reason a reader then goes looking for.
- **A conditional value is `wp.where(cond, a, b)`, not a Python ternary** (check 20). A ternary
  compiles to the same code — Warp lowers `ast.IfExp` exactly as it lowers `wp.where` — so like
  checks 16, 17 and 18 this is legibility, and nothing but a scan holds the line. Note `wp.where`
  eagerly evaluates both arms where a ternary short-circuits; every kernel-scope ternary found to
  date has both arms already evaluated, so a future site where they are not needs a decision rather
  than a mechanical conversion.

Checks 16, 17, 18, 20 and 22 are **one family**, and the family is the point: every one of those
spellings is *legal* and generates identical code, so the defect is invisible to the compiler and to
the suite, and nothing but a scan holds the line.

### 1.6 Mutable loop accumulators

Inside `@wp.kernel` / `@wp.func`, a variable initialized to a **bare numeric literal**
(`count = 0.0`, `n = 0`) is a **compile-time constant** and cannot be mutated inside a dynamic
`for` / `while` loop. Warp raises `WarpCodegenError: "Error mutating a constant count inside a
dynamic loop…"` — or, in some module contexts, **silently keeps the initial value**, so `mean /
count` divides by zero and produces `inf` with no error.

Declare mutable accumulators with the typed constructor: `count = wp.float32(0.0)`,
`n = wp.int32(0)`. (The `float(0.0)` / `int(0)` spellings work identically — §12.5 — but §1.3's cast
rule means the tree writes `wp.float32` / `wp.int32`.)

**Never let a ruff auto-fix strip the constructor off a kernel-scope accumulator** — it introduces a
real bug. Where the legacy `int(0)` spelling survives, **two rules fire, not one**: `float(0.0)`
trips only `UP018`, but `int(0)` also trips `RUF046` and `# noqa: UP018` alone does not suppress it
— `ruff check --fix` silently rewrites it to `0` and the kernel then fails to compile on the next
launch.

---

## 2. Kernel architecture and launches

### 2.1 Launch discipline

- `wp.launch(kernel=..., dim=..., inputs=[...], device=...)`. **Always forward the `device` from the
  input arrays** — this is a memory-safety rule, not tidiness (§3.9). Check 15 scans every launch
  site; every launch in `triwarp/` names a device.
- Array slicing is supported inside kernels: `faces[f * 3 : (f + 1) * 3]` produces a sub-array view
  (but see §1.4 — prefer an index form where one exists).
- **Prefix output argument names with `out_` and put them at the end of the signature**, after all
  inputs (check 13). Two exemption classes, both carried as `_KERNEL_OUTPUT_ALLOWLIST`:
  **in-place** arguments, where the same buffer is input and result — an `out_` prefix would misread
  as write-only; and **scratch / persistent-state** buffers, caller-allocated working memory carried
  across launches (cursors, stacks, open-addressing tables, `ball_pivoting`'s front) — neither an
  input nor the answer, so name them for what they hold. A read-only input must never wear the
  `out_` prefix, even when the buffer was a *producer* kernel's output — parameter names describe
  the argument's role in *this* kernel.
- Derive the face count as `f = faces.shape[0] // 3` from the flat face index array.

### 2.2 A lane-parallel kernel strides by `wp.block_dim()`, or it stays lane-free

A kernel whose lanes cooperate — a `wp.tile_sum` / `tile_min` / `tile_max` over one value per lane,
a `wp.tile_bvh_query_aabb` walk — is correct on **both** devices exactly when its lanes partition a
sequence **the block already owns**, with the stride taken from `wp.block_dim()`. Never from a
kernel argument, never from a module constant.

`wp.launch_tiled` runs exactly **one lane per block on the CPU device** through Warp 1.17 whatever
`block_dim=` is passed, and `wp.block_dim()` reads `1` there — so that single lane walks the whole
sequence and the one-element tile it reduces holds the right answer. **A one-element tile is not the
bug.** An arg-strided sum instead is wrong on **both** devices: on CPU the single lane walks only
its stride's share (short by exactly the stride), and on CUDA the answer is correct only when the
wrapper happens to pass `n_slices == block_dim`, which no signature expresses — pass a larger
`block_dim` and the surplus lanes re-walk what the first ones counted.

**Do not read a CPU failure of a tiled kernel as "tiles do not work on the CPU backend."** That
reading cost this package four kernels converted *away* from tile reductions and four comments
stating the prohibition in general terms, all of which the block-per-item kernels then contradicted
correctly. Full platform detail and the probe that distinguishes the two cases: §12.2.

Where the lanes would partition the **outer** work the grid is over — a whole-array reduction with
no per-item dimension, `measures.centroid_tiled`, `metrics.chamfer_*_tiled` — there is no
block-owned sequence and no `wp.block_dim()` to take. Those kernels must either keep a *constant*
stride and be launched on CUDA only, with a lane-free `_sliced` sibling for the CPU (the
`_device.prefers_tiled_reduction` pair), or stay lane-free on both. Both are correct; a single
kernel is not available. **Rule of thumb: a strided loop is one token from portable; a block-index
partition is a rewrite.**

Read `kernels/visibility.py::obscurance` (one block per point, lanes over its ray bundle, measured
3.2-11.8x against one thread per point) for the first form and `kernels/measures.py::centroid_tiled`
for the second.

**A lane-strided reduction's tuning variable is the fold width, not the lane redundancy.** Launch it
`dim=kernel_reduce.blocks_1d(n)` so each block owns `ITEMS_PER_BLOCK_1D` elements, never
`dim=n / TILE_1D`: what reaches a constant-slot accumulator is *one atomic per block*, so the block
count is the contended quantity and the fold is what pays for the `wp.tile_sum` calls. Two kernels
that already had one atomic per block, and looked finished for that reason, measured **4.1-10.7x**
from the fold alone — while removing their 64-fold redundant lane arithmetic, the thing that *looks*
wasteful, was worth 1.02-1.13x. The wide-accumulator caveat and the commit spelling (`wp.atomic_add`
reads trailing indices as array dimensions, so assemble the summed vector or matrix first): §13.2.

### 2.3 Block-per-item is an occupancy trade, not a shape choice

**Eligibility is an occupancy question, and getting it backwards costs 2-8x.** Having an outer
per-item dimension is necessary and not sufficient: the block-per-item form pays exactly when that
outer dimension *alone* would starve the device. `obscurance` qualified because it launched
`dim = n_points` — a few thousand threads, under 3 % of the device — with no second dimension at
all. A kernel that already carries a **slice dimension** does not qualify, because that dimension is
what fills the device and collapsing it into `block_dim` lanes throws the occupancy away.

Measured on `points.hull_support_extremes`, converted both ways: at 5 000 points (13 x 40 threads)
block-per-direction is **2.3x faster** because the grid was starved; at 200 000 (13 x 1 563 threads)
it is **0.12-0.60x — a 2-8x loss**, 13 blocks on 170 SMs.
`proximity.winding_number_tiled` shows the same trend from the other end: 2.16x at 1 280 faces and
4 096 queries, 1.19x at 5 120 faces, **1.01x** at 65 536 queries — a gain that shrinks to nothing as
the grid fills, which §9 calls a decline rather than a small win.

**Before proposing this rewrite, compute the launch's *current* thread count**, and measure at the
large end first — the small end shows a flattering 2x. Four kernels in this tree keep the
arg-strided form deliberately and each says so with its number. The full win/loss table is §14.1.

### 2.4 Fusing a kernel: extract the shared part as a `@wp.func` in the same commit

Fusing two launches into one is a standard and welcome optimization here — it removes a launch, a
round trip through global memory, and often an intermediate buffer. But a fused kernel is written by
*copying* the prologue of the kernel it absorbs, and the copy is what survives. **A fusion is not
done when it is faster; it is done when the code it duplicated has a name.**

Whenever a kernel is fused, split, specialised, or given a second variant (tiled/serial,
float32/float64, one accelerator/another), the same commit must:

1. **Name the shared run.** Any consecutive statement run the new kernel shares with the one it came
   from — a corner-index load, a window computation, an emit protocol, a guard sequence — becomes
   one `@wp.func` that both call. `@wp.func` calls are inlined at codegen, so this costs nothing at
   runtime; it is free structurally and the only reason not to do it is that nobody looked.
2. **Put it where the *quantity* lives, not where the fusion happened.** A general geometric
   predicate goes in `kernels/predicates.py`, a per-face quantity in `kernels/triangles.py`, an
   index/sort/search helper in `kernels/array.py`, a scatter in `kernels/scatter.py` (§3.1 names
   these four and why). A shared helper left in the algorithm module is how `triangle_aabb`,
   `triangle_double_area` and `circumcircle_diameter` ended up being reached by unrelated modules
   importing an algorithm to get at geometry.
3. **Say in a comment what the two kernels still differ by**, so the next reader can tell a real
   variant from a stale copy. Where the difference is a *parameter* rather than an algorithm, prefer
   one kernel with a warp-uniform int selector (§2.7's `wp.Function`-as-argument restriction and the
   `ACCEL_HASHGRID` / `ACCEL_BVH` pattern in `kernels/neighbors.py`); where it is genuinely two
   algorithms, keep two kernels and have each name the other.

**Merge on identity of meaning, not identity of tokens.** Two bodies that agree because they compute
the same quantity are one function; two bodies that agree because a one-line kernel has only one
shape are two functions and the duplicate scan's hit is noise. `remesh.compute_midpoints` and
`triangles.face_centroids` normalise identically and must stay apart.

**A comment that names the other copy is the finding, not the fix.** *"Feature handling mirrors
`collapse_candidates`"*, *"as in X"* — each of those was written by an author who had already seen
the duplication and answered it with prose. A cross-reference is a claim that **only a shared
`@wp.func` can keep true**. So when writing "same as X" / "mirrors X" about *code* rather than about
a reason, extract instead; and when reading one, treat it as an unfactored duplicate that has been
located for you. **Re-count the run from the code, not from the comment** — one "same two lines" was
seven statements, and the prose had been factored where the code had not.

**A clean duplicate scan is not a clean file.** Every scan this package has used keys on α-renamed
statement *text*, so two bodies expressing the same decision through different control flow — a
`reject` flag against an early `return`, a position against a boolean — score as unrelated. **A
duplicated *decision rule* is a live correctness hazard where a duplicated arithmetic run is only
noise**, so it outranks the longer runs a scan does find: extract it as a `@wp.func` returning the
classification (a sentinel for "reject", the `_resolve_flip_quad_guarded` convention) and let each
caller map the free branch onto its own answer.

**Factor the family, not the pair — a helper pinned to one rank or one precision breeds the copies
it was meant to prevent.** Two instances: `corner_triple` landed for flat 3-stride buffers and
reached 16 callers while its rank-2 sibling (`arr[row, 0..2]`) stayed nameless at eight sites; and
`laplacian.squared_edge_lengths`, hardcoded `wp.vec3` / `float32`, could not be reached by the
`float64` sites in `energies.py`, which open-coded it instead. When extracting, write the generic
form §1.2 asks for and check for the *other* rank and the *other* precision before declaring the run
named. A general per-triangle quantity left in an algorithm module is a **three-time** defect, and
the tell is a kernel module importing an *algorithm* to reach *geometry*.

**One predicate, one spelling per module — this is a correctness rule, not a style one.**
`wp.length(d) < r` and `wp.length_sq(d) < r * r` are **not the same predicate in float32**:
measured, 10 rows of 200k disagree at the boundary. So a module that tests the same rule both ways
can accept a candidate in one kernel and reject it in another. Pick one spelling per predicate and
say which; expect the speed to be flat (measured 0.997-1.003x — §13.2) and keep the number either
way.

**A green suite does not prove a `@wp.func` extraction was behaviour-neutral.** `@wp.func` calls
inline at codegen, so an extraction that reorders an expression's evaluation changes `float32`
results without moving any comparison asserted at `1e-5`. The evidence is **reading the diff**, not
the suite. Two consequences for the gate: prefer to check the *decision* arrays a kernel writes over
the positions it produces where positions drift on their own (§16.4), and where a reference's answer
is a combinatorial object — a triangulation, a face buffer — gate on that rather than on a
tolerance.

Watch the signature while fusing: a fused kernel inherits the union of two argument lists, and a
launch argument costs ~1.0 µs of host time (§2.8, §13.1).

**And price the fused region itself, never the call that contains it — §14.10 is the verdict
table.** Every such pair measured in this tree is faster fused (1.04-3.46x); what the iteration
count changes is only whether a benchmark can resolve the win. Attributing through the enclosing
call declined a real `heat` win on noise once already.

**The converse of rule 1: a helper with one caller is a name, not an abstraction — inline it, then
look again.** Extracting the shared run is right when the run is *shared*; when a fusion absorbs its
own producer, the "helper" it leaves behind has a single call site and its only effect is to hide
the redundancy between the two halves. Inlining three such kernels exposed a doubled vertex load, a
doubled grid index and a `dbl_area` computed twice by two spellings of the same expression —
together worth more than the launch the fusion removed (§14.10).

### 2.5 A generic kernel registers its overloads at import (`wp.overload`)

**A `@wp.kernel` generic over a dtype (`wp.Scalar`, `wp.Float`, `wp.Int`, `Any`) must have its
concrete overloads registered at module import**, in a `_register_overloads()` called at the bottom
of the file. Warp instantiates a generic kernel's overload *lazily*, on the first launch at each new
dtype, and a module's hash covers the set of **instantiated** overloads — so that first launch
changes the hash and recompiles **every kernel in the module**. Nothing fails; it only costs, which
is why nothing but a check catches it. Before the registrations existed, `triwarp.kernels.*` rebuilt
across 206 module loads in a full suite run against 87 after (2 per module is the irreducible
CPU/CUDA floor), and one test file went from ~9 minutes to 1.4 s.

- **The chain is order-dependent**, which makes it a developer-loop tax rather than a one-time cost:
  a caller reaching the dtypes in a different order walks links that were never compiled, so
  changing which tests you select re-pays it from scratch.
- **Register what the wrapper's dispatch can reach, not every dtype the template admits** (§4.2, no
  speculative generality): an unused overload is compile time paid on every rebuild. Derive the set
  from the wrapper — a public `dtype=` keyword documented "float32 or float64", the key dtypes
  `sortable_dtype` maps onto, a docstring naming its own admissible dtypes — and say so in a
  comment. Where two generic arguments are independent (`laplacian.cotmatrix_triplets`' entry
  precision and matrix precision), it is a genuine cross product, not a diagonal.
- **A diagonal registration where the wrapper admits a cross product is the expensive half of this
  rule.** `energies.crouzeix_raviart_cotmatrix_triplets` was registered with one loop variable
  serving both its `cot_entries` and its value precision, while the public wrapper takes the two
  *independently* and the kernel body casts one to the other; the first off-diagonal call then
  logged `Module hash changed, recompiling` and took over a minute where the nested loop takes
  milliseconds. So when a wrapper exposes two dtype knobs, check whether the registration crosses
  them; `laplacian.py`'s sibling does and says so in a comment, which is the model.
- **Registration is not compilation.** `wp.overload` builds the overload's `Adjoint` and nothing
  else, so this costs milliseconds of import and nothing on a process that never launches the
  kernel. Do **not** `wp.load_module` / `wp.force_load` at import — that *would* compile eagerly.
- **`block_dim` forks the hash independently of dtype and is normally left alone.** Its values are
  fixed by the package's own launch code (a bounded two or three: a tiled launch's `block_dim`,
  Warp's 256 default, and 1 on CPU), so it is not a chain that grows with call order. Collapsing
  them is a perf change and needs §9's measurement — it was measured for `reduce` and declined.
- `tests/test_api_conventions.py::test_generic_kernels_register_their_overloads` fails when a
  generic kernel has **no** overload registered. Nothing checks that a registered dtype *set* is
  complete, and nothing cheaply can. **A missing dtype is diagnosed from the clock, not from a
  failing assert** — see §15.1.
- **`wp.map` has the same chain**, keyed by the unqualified op name and forked per *signature*, and
  it needs the same treatment — §3.5 (check 23) and §12.6.
- **And the registration pays twice: keep the `wp.Kernel` `wp.overload` returns, and launch through
  it.** Registering fixes the *rebuild*; it does nothing about the ~12 µs of `infer_argument_types`
  every generic launch runs before it can even look the overload up (§13.1, measured 2.17x on one
  launch and 1.09-1.58x end to end across 20 wrappers). So a module's registration builds a
  dtype-keyed [`OverloadTable`][triwarp.kernels.array.OverloadTable] and the wrapper writes
  `kernel_laplacian.COTMATRIX_TRIPLETS[cot_entries.dtype, dtype]` rather than naming the generic
  kernel. Nothing extra is compiled, and the kernel source stays generic, so §1.2 is untouched.
  **All generic launch sites in the tree are converted**; a new generic kernel adds a table rather
  than a bare `wp.overload` call. A dtype the table lacks raises a `KeyError` naming the kernel,
  which is this section's failure mode made visible instead of costing a silent rebuild.
  `kernels/reduce.py` goes one step further and has **no** generic kernels at all: its factories
  always could bake the dtype in, and its own `_reduce_1d_tiled` docstring had said so since it was
  written — the ones that stayed generic were simply never revisited.

### 2.6 In-place `@wp.func` parameters (`wp.ref[T]`)

`@wp.func` helpers may declare `wp.ref[T]` parameters to mutate caller-owned storage (locals, array
elements, struct fields) — use for multi-value updates like argmin/minmax/swap helpers (see
`update_argmin` in `kernels/array.py`).

- **Constraint:** any kernel calling a `wp.ref` helper must be decorated
  `@wp.kernel(enable_backward=False)` — the per-kernel flag specifically; a module-level
  `wp.set_module_options({"enable_backward": False})` is NOT consulted at kernel-parse time and the
  module still fails to compile. This is inference-neutral (no forward-runtime effect).
- **Never use `wp.ref` in `triwarp/kernels/metrics.py`** — the chamfer kernels are differentiated
  via `wp.Tape`.
- **`wp.ref[T]` requires a concrete `T`** — generics do not instantiate inside it. Details and the
  knock-on consequences: §12.3.

### 2.7 Function-valued parameters and kernel factories

- A `@wp.func` may take `fn: wp.Function` parameters; the target is bound at **compile time** per
  call site (user `@wp.func`s and simple builtins like `wp.min` are valid targets; tile intrinsics,
  variadic and LTO builtins are not).
- **`wp.launch` can NOT pass a `wp.Function` as a kernel argument.** For runtime selection, pass an
  **int/enum kernel argument** and branch over `wp.Function` targets inside a dispatch `@wp.func`
  (warp-uniform branch, one compiled module; see `registration.robust_weight`).
- Builtins with no Python-scope handle (e.g. tile intrinsics) can still parameterize kernel
  factories by pulling the concrete `Function` object from
  `warp._src.context.builtin_functions["tile_max"]` and closure-capturing it — captured builtins
  emit inline at codegen and template on the tile dtype (see `triwarp/kernels/reduce.py`). Give each
  factory instantiation a unique kernel `name`.
- **Name it with `wp.kernel(f, name=...)`, not by mutating `__name__` / `__qualname__`.** Warp 1.17
  added a `name=` parameter to the decorator (NVIDIA/warp#1561) that sets both the registration key
  and the base of the generated native entry point; it must be a valid C++ identifier. That is one
  line instead of two assignments the reader has to recognise as a naming convention, and it is the
  documented route to ahead-of-time compilation.
- **A factory is also how you avoid a generic kernel's dispatch cost.** A generic kernel pays
  ~12 µs of host-side overload resolution on *every* launch — the same for `Any`, `wp.Float` and
  `wp.Scalar`, and more when several parameters are generic (§13.1). A factory that writes
  `__annotations__` after defining the body produces concrete kernels from one source:
  ```python
  def _factory(name, dtype):
      def _k(values: wp.array[wp.Scalar], out: wp.array[wp.Scalar]) -> None: ...
      _k.__annotations__["values"] = wp.array[dtype]
      _k.__annotations__["out"] = wp.array[dtype]
      return wp.kernel(_k, name=name)
  ```
  **A factory whose `dtype` parameter has a generic default is a factory nobody specialised**, which
  is exactly how `kernels/reduce.py` ended up with dozens of generic kernels behind a template that
  could always have baked the dtype in. If a factory takes a dtype, give it no default.
- **Where the kernel body should stay generic, the table form in §2.5 gets the same launch cost for
  no restructuring at all** — `wp.overload` already returns the concrete kernel. Reach for a factory
  when the *body* needs specialising (an axis, a storage class, a captured builtin) and for the
  table when only the dtype does.
- **`if wp.static(flag):` prunes the untaken branch even when the two branches bind locals of
  *different types*** (probed: `wp.types.vector(length=K)()` vs `wp.zeros(shape=K)`, and
  `values[i, k]` vs `values[k, i]`). That is what lets one body generate axis- or
  storage-specialised kernels with no runtime branch.

### 2.8 `@wp.struct` argument bundles

**A `wp.launch` argument costs ~1.0 µs of host time, linearly, on both CUDA and CPU** (§13.1). §9's
flat "~32 µs per launch" is the *mean* kernel's launch (mean argument count 5.1), not a constant.

If a kernel is launched inside a Python loop and carries a dozen or more arguments, bundle the
invariant tables into a `@wp.struct` built **once** in the wrapper — measured 43 → 18 µs per launch
for a 25-argument kernel, a flat saving at every `dim`, with subscript-style field annotations
(`a: wp.array[wp.int32]`) and 2.6 µs per bundle construction. Rebuilding per launch gives the win
back.

- **Width alone does not qualify a kernel — the launch count around it does.** Count the launches
  per call and divide. `remesh.objective_flip_candidates` is the same 14 arguments and is declined:
  a flip *round* rebuilds the whole face adjacency around its one launch and converges in **2**
  rounds, so nine bundleable arguments bound the saving at under 2 % of the call.
- **Do not bundle a kernel whose launch is already captured** — a replayed launch costs ~1.17 µs
  whatever its argument list (§13.1), so `remesh._issue_pass` and the multigrid V-cycle inside
  `wp.capture_while` gain nothing.
- Do **not** open a tree-wide bundling pass: only 18 of 440 kernels take ≥12 arguments, and §4.2's
  "no speculative generality" applies to argument bundles too.
- **A new Warp construct can switch off a static check written before it.** Moving buffers into a
  `@wp.struct` removed them from check 13's view entirely, because the check resolved store targets
  to a bare `ast.Name` — see §4.5.

Two examples worth reading before writing a third: `holes._fill_dp` launches a 16-argument kernel
once per span, and `reconstruction._bpa_wave` launches a 27-argument one per wave. Measured gains
and the graph-capture comparison: §13.1, §14.3.

### 2.9 Per-thread row storage

**`wp.zeros(shape=K, dtype=T)` inside a kernel is not per-thread fast storage.** Warp wraps the
stack buffer in an `array_t`, so every access goes through a pointer and nvcc never promotes it to
registers — measured a **2x loss** against a global-memory row at k=32.

Use **`wp.types.vector(length=K, dtype=T)`** (the `vec5d` pattern in `kernels/curvature.py`):
genuinely register-resident, 1.09x at k=1, 1.5x at k=7, 1.9x at k=16, 2.9x at k=30, **7.8x at
k=64**, collapsing to 1.66x at 96 where it spills (the row costs `2 * K` registers).

**Keeping it in registers constrains the code shape:**

- A vector must never be passed to a `@wp.func` (that takes its address and spills it) — the insert
  has to be inline, duplicated per call site.
- **Any runtime index spills the whole vector**: read the k-th element with an unrolled
  `for slot in range(K): if slot == k - 1:` compare, not `row[k - 1]`. `K` is closure-captured in a
  kernel factory so `range(K)` unrolls with literal indices; one kernel per bucket, unique `name`
  each (`KNN_ROW_BUCKETS = (1, 4, 8, 16, 32, 64)`).
- **A register insertion sort needs an explicit `placed` flag.** Carrying the displaced element down
  with `if carry < row[slot]` breaks the chain on a run of **equal** distances already in the row,
  dropping a neighbour — one differing row in 20 000 at k=32 with the distance array still
  byte-identical, i.e. invisible to a distance comparison.

Cost of admission: `kernels/neighbors.py`'s cold-cache compile roughly tripled for 12 generated
kernels. Related:

---

## 3. Python-scope wrappers

### 3.1 Module layout and imports

- Every kernel lives in a `kernels/` sub-module, imported with an alias:
  `from triwarp.kernels import triangles as kernel_triangles`.
- **A top-level kernel module is named exactly for the public module it backs**:
  `triwarp/kernels/<module>.py` ↔ `triwarp/<module>.py`, one-to-one (check 7). The only admissible
  exceptions are the kernel-side libraries that back no single public module —
  `kernels/predicates.py` (geometric `@wp.func` predicates) and `kernels/scatter.py` (scatter /
  accumulate kernels). Sub-packages (`kernels/algorithms/`) mirror a folder rather than a module and
  are exempt. Adding a kernel module with no public counterpart, or a public module whose kernels
  live under another name, is a defect — fix the name, do not document the exception.
- **Backing a public module and serving as a shared library are not exclusive**, and four modules do
  both jobs. Being imported across the tree is not a violation and needs no exception; it is what
  these four are *for*:

  | Module | Holds |
  |---|---|
  | `kernels/array.py` | index/sort/cast/search `@wp.func`s (`sort3`, `cross2`, `to_vec3d`, `binary_search_index`) |
  | `kernels/predicates.py` | precision-generic geometric predicates |
  | `kernels/triangles.py` | per-face corner/quality/gradient `@wp.func`s |
  | `kernels/scatter.py` | scatter/accumulate kernels |

  What *is* a defect is placement: a general geometric predicate living in a module that owns an
  **algorithm**, so that unrelated modules import the algorithm to reach the geometry.
  `triangle_aabb` sat in `kernels/intersection.py` and `triangle_double_area` /
  `circumcircle_diameter` in `kernels/holes.py` for exactly that reason; all three are now in
  `predicates.py`, generic over the scalar type. When a helper is reached from a second module, ask
  which of the four it belongs in before adding the import.
- **`triwarp/__init__.py` is lazy (PEP 562 `__getattr__`) and must stay that way.** `@wp.kernel`
  builds an `Adjoint` at *import* time, so an eager `__init__` made `import triwarp as tw` a
  whole-package pull; see §16.2 for the measurement and the subprocess test that guards it.
- **Three searches in `kernels/array.py` are not interchangeable**, and picking wrong is silent:
  `binary_search_index` is `searchsorted(side="right")` (returns `slot + 1` on an exact hit),
  `binary_search_index_left` is `side="left"` (the exact slot for a present key — use this plus
  `index < n and values[index] == v` for a lookup), and `binary_search_sorted_contains` is
  membership only. Detail and the bug it caused: §16.5.
- **A radix sort of packed keys whose radix is known passes `end_bit`**, and the producer writes the
  keys straight into the sort's double-width buffer. `wp.utils.radix_sort_pairs` otherwise orders
  every bit of the key, at a cost that is mostly fixed per digit pass, so a 64-bit sort of edge
  keys below `n_vertices ** 2` pays for passes over bits that are always zero -- measured 2.1x on the
  sort at 24 k keys (§13.1). The permutation is identical (the sort is stable), so it is
  byte-identical on both devices. `unique_1d(max_value=)`, `adjacency.sorted_face_edge_keys` and
  `unique_faces(max_index=)` carry the bound through; the pair radix (`INDEX_RADIX_PAIR`) needs all
  64 bits and gains nothing.

### 3.2 Typing (`triwarp.typing`)

Import once per Python wrapper module:

```python
import triwarp.typing as twt
```

Do **not** re-export typing symbols from `triwarp/__init__.py`; import `twt` where needed.

**Why not `wp.array2d` in wrappers?** At runtime every buffer is `warp.array`. `wp.array2d[dtype]`
in a Python signature is a static annotation helper; type checkers do not treat it like a real array
(missing `.shape`, bad assignability from `wp.empty`), and `isinstance(x, wp.array2d)` is always
`False`.

Use **`wp.array[dtype, Literal[ndim]]`** via the aliases:

| Alias | Meaning |
|-------|---------|
| `twt.Array2dInt32` | `(rows, cols)` `int32` |
| `twt.Array2dFloat32`, `twt.Array2dFloat64` | `(rows, cols)` `float32` / `float64` |
| `twt.Array1dInt32` | 1D `int32` |
| `twt.IntArray`, `twt.FloatArray`, `twt.ScalarArray` | 1D or 2D unions (e.g. `reduce.py`) |

Kernels in `triwarp/kernels/` keep `wp.array2d[dtype]` unchanged. Optional 2D arguments:
`edges_sorted_wp: twt.Array2dInt32 | None = None`.

**Runtime checks, not `isinstance`:**

- `twt.ensure_ndim(arr_wp, 2, dtype=wp.int32)` — validate rank and dtype on inputs.
- `twt.as_array2d(arr_wp, wp.int32)` — check then narrow the return type for Pyright. Overloaded on
  the `dtype` argument for `wp.int32` / `wp.float32` / `wp.float64`; `twt.as_array3d` covers
  `wp.float32` / `wp.bool` at rank 3.

### 3.3 Allocation and returns

- Python-scope wrappers accept `wp.array[T]` for 1D buffers; use `twt.Array2dInt32`,
  `twt.Array2dFloat32`, etc. for rank-2 results.
- For **2D** outputs allocate with `twt.empty_2d((rows, cols), wp.int32, device=...)` — the dtype is
  an argument, not part of the name. `twt.empty_3d` is the rank-3 counterpart.
  **`empty_1d` / `empty_2d` / `empty_3d` are one allocator at three ranks: each carries its
  `dtype` argument straight into its return type through a `TypeVar`, so every Warp element type
  works at every rank and there is no per-dtype overload table to extend.** The same holds for
  `as_array2d` / `as_array3d`. **Do not answer "this allocator does not accept my dtype" by adding
  an overload** — if one is ever needed, the reason will be a *runtime* restriction, and it belongs
  in the shared `_empty_ranked` body with the rest of the validation.
- For **1D** outputs keep `wp.empty(n, dtype=..., device=input.device)` when all elements will be
  written by the kernel (avoid unnecessary zero-initialization). **`twt.empty_1d(n, dtype,
  device=...)` is for the modules whose signatures carry the rank** — `reduce`, `metrics`,
  `neighbors`, where a `k=1` or an `axis=` collapses a rank and so the `twt.Array1d*` aliases
  are load-bearing. It is not a general replacement: `wp.array[dtype]` is this package's usual
  rank-1 spelling and `wp.empty` already satisfies it, so reaching for `empty_1d` there buys
  nothing and, because `NDim` is **invariant**, actively breaks the assignment (§8).
- Return rank-2 buffers as `return twt.as_array2d(arr, wp.int32)` so callers get a checked,
  correctly typed value.
- **Always pass `device=` to every allocation** (`wp.zeros` / `empty` / `ones` / `full` / `array`) —
  check 10, and it scans all of `triwarp/` including `_*.py`. Without it the buffer lands on Warp's
  *current* device and the suite cannot see the difference, because a test runs with its arrays'
  device already current: `array.index_sparse`'s `wp.ones` was wrong for as long as it existed and
  every test passed.
- Empty mesh / no adjacency early return:
  ```python
  if n_faces == 0:
      return twt.empty_2d((0, 2), wp.int32, device=faces_wp.device)
  ```
- **Size buffers for their final use at allocation time.** Do not allocate-then-grow at Python
  scope. When a consumer needs an `n + 1` sentinel-terminated form, the *producer* allocates `n + 1`
  and hands back a view (`counts_to_offsets`); a helper whose only job is to patch up another
  function's output convention is a smell to be fixed at the producer.
- **Warp raises on a zero-length slice** (`RuntimeError: Invalid indexing in slice: 20:20:1`), so a
  trailing-mask `fill_` needs an `if stop > start` guard where a NumPy version silently no-opped.
- **A buffer whose initial value matters is *allocated holding it* — never `wp.empty` followed by
  `fill_` / `zero_`.** One statement instead of two, and no window in which the buffer holds
  garbage. `wp.zeros`, `wp.full(n, value, dtype=..., device=...)`, and at rank 2
  `twt.as_array2d(wp.full((rows, cols), value, ...), dtype)` — there is deliberately no
  `twt.full_2d`, because that spelling already appears at the one site that needs it and §4.2
  forbids the speculative helper. **The reason is legibility, not speed**: `wp.empty` plus `fill_`
  against `wp.full` measures 0.96-1.00x at every size and rank, so the two-call form is if anything
  a few tenths of a microsecond *cheaper* and the choice is entirely about how the code reads.
    - `wp.empty` stays correct — and is the rule — where **every** element is written before it is
      read. What this rule forbids is allocating uninitialized and then initializing.
    - **Where the branches initialize differently, allocate inside each branch.** A shared
      allocation above a branch that initializes reads as one thing and is as many things as there
      are branches.
    - **A *partial* write into a buffer another writer already filled is not this pattern.** Ten
      `arr[a:b].fill_(...)` sites carry a running state counter, a padded triplet index, or a
      mask whose head a kernel wrote; those stay. The scan that finds the real thing keys on a
      whole-buffer `fill_` / `zero_` on a name assigned from `wp.empty` / `twt.empty_*` within a
      few lines, and a second pass keying on a *slice* target finds one more, so run both.
    - There is **no** Python-scope scalar write to pair with `_device.read_scalar`, and asking for
      one is usually the wrong question: `arr[k] = v` raises `TypeError: 'array' object does not
      support item assignment` on both devices and `arr[k : k + 1].fill_(v)` is already the
      primitive such a helper would wrap. The site that prompted the question wanted the
      *allocation* to carry the value instead, after which the write disappears.

### 3.4 Python-scope gather indexing (prefer over trivial gather kernels)

At **Python scope**, Warp supports **gather** with integer indexing: `view = src[indices]` yields a
`wp.indexedarray`. Materialize a dense `wp.array` with `wp.copy(dst, view)` when callers need
`.reshape()` or a guaranteed `wp.array` return type (see `triwarp/selection.py`'s face gather and
`triwarp/array.py`'s `isin`).

- **1D gather:** `vertices[indices]`, `lookup[elements]`.
- **2D index arrays:** Warp requires **1D** index arrays — flatten first (`elements.flatten()`),
  gather, `wp.copy`, then `.reshape(original_shape)`. `.flatten()` on rank-3/rank-4 arrays returns a
  **contiguous** rank-1 view and `reshape` round-trips fine, which is why `array.isin` accepts any
  rank instead of capping at 2. But `wp.array.flatten()` **raises** `RuntimeError` on a
  non-contiguous view rather than copying.
- **⚠️ The index array must be CONTIGUOUS.** Warp reads the index buffer as if contiguous and
  **silently ignores a view's stride** — no exception. `payload[edges[:, 0]]` returns the flattened
  buffer's leading entries (`[0, 10, 1, 11, …]`), not column 0. A contiguous *prefix* slice
  (`arr[:n]`) is safe; a column (`arr[:, k]`) or step slice (`arr[::2]`) is not — `wp.copy` it into
  a dense buffer first. **Both halves of that sentence are load-bearing, and the tree got the first
  one backwards once**: a `gather(eligible, wp.clone(order[:k]))` cited this rule where `order[:k]`
  is a prefix and needs no clone at all — dropping it was worth 1.50x, byte-identical. When a
  clone-of-slice cites this rule, check which kind of slice it is; `adjacency.face_adjacency` and
  `reconstruction._seed_candidates` clone a `[:, 0]` column and are the case it exists for.
  Measurements: §12.1. This is why `kernels/edges.py:edge_lengths` stays a kernel. **When converting
  a gather, verify values, not just that it runs and is faster**: the corrupt version reads a
  contiguous prefix and is measurably *faster* than the correct one.
- **Indexed assignment** (`arr[indices] = value`) is **not** supported on `wp.array` at Python scope
  — keep a small kernel for scatter / mask marking (e.g. `mark_membership_mask` in
  `triwarp/kernels/array.py`).

Do **not** add custom per-element gather kernels when `[]` plus `wp.copy` suffices. Probe tests live
in `tests/test_*_indexing_probe.py` — `test_array_indexing_probe.py` pins the stride hazard
itself (still live on Warp 1.17), written as *"the gather returns the flat prefix"* rather than
*"not the column"* so that a Warp release honouring the stride fails it and this rule is revisited
rather than silently kept.

### 3.5 Elementwise ops at Python scope (`wp.map`)

Do **not** write a `@wp.kernel` whose body is only `out[i] = f(in[i], ...)`. Keep the op as a named
`@wp.func` in the `kernels/` module (or use a builtin like `wp.neg`, `wp.add`, `wp.div`,
`wp.normalize`) and call **`wp.map(op, *inputs, out=...)`** from the wrapper. The generated kernel is
cached (in-memory per process + Warp's on-disk cache) and its GPU time is identical to a
hand-written kernel; cached calls cost ~11 µs extra host-side Python.

- **Named `@wp.func` only, never lambdas**: the map cache is keyed by the *unqualified* function name
  plus input dtypes — two different ops with the same name would collide, and lambdas re-derive the
  function each call.
- **Always pass `out=`** so allocation stays with `wp.empty` / `twt.empty_*` in the wrapper. In-place
  is `out=<an input>`; multi-output funcs (`tuple[...]` return) take `out=[a, b]`.
- Scalars mix freely with arrays (`wp.map(is_long_edge, lengths, max_edge_f, out=mask)`); device is
  inferred from the array inputs. **But `wp.map` infers a bare Python `int` as `wp.int32`, not as the
  mapped func's declared dtype** — unlike `wp.launch`. A `bvh.id` handed to a `wp.uint64` parameter
  fails at codegen with `TypeError: Function <fn> does not support the provided argument types
  int32, ...`. Cast explicitly: `wp.map(fn, wp.uint64(bvh.id), ...)`.
- **Slice views** work as inputs and outputs: adjacent-element ops map over shifted views
  (`wp.map(segment_length, polyline[:-1], polyline[1:], out=lengths)`), and offset writes map into
  `out=dst[o : o + n]`. CSR row degrees: pass `offsets[:-1]` and `offsets[1:]`.
- **Python-scope gather composes**: `wp.map(pred, table[indices], out=mask)` maps over the
  `wp.indexedarray` view (see `repair.make_volume`).
- Inside **per-iteration wrapper loops**, hoist the kernel once with `wp.map(..., return_kernel=True)`
  and `wp.launch(kernel, dim, inputs=[...], outputs=[...])` in the loop (see `triwarp/smoothing.py`)
  — this removes the per-call Python overhead, and the overhead has a number: **a cached `wp.map`
  call costs 1.78-1.86x the launch it wraps, ~11 µs**, flat in the array size. It is the same
  host-side resolution a generic kernel pays (§13.1), so the two conversions look alike and are
  priced alike. **Hoist where the loop body is cheap; a map inside a solve loop is a fraction of a
  percent** — `linalg._multigrid_hierarchy`, the ARAP and CG component loops and the remesh pass
  loops are all left alone deliberately.
- **An op reached at more than one call signature must have those signatures declared at import**
  (check 23) — the same rule and the same reason as `wp.overload` in §2.5. `wp.map` names its
  generated module `map_<unqualified op name>`, each distinct signature forks that module's hash, and
  a module's hash covers the kernels instantiated in it — so an op reached at three signatures builds
  its module three times, each build containing every kernel accumulated so far. Over the tree's
  eight longest chains that is **2.1x** of cold-cache compile time and 1.18x warm.
  **Declaration is not compilation** — `return_kernel=True` on a zero-length host array is
  sub-millisecond and reaches the final module directly. The tables live in `_declare_map_kernels()`
  at the bottom of the module that owns the op (`declare_map_signatures` in `kernels/array.py`
  carries them and the reasoning); the Warp *builtins* live in `kernels/array.py` because one
  generated module is shared across several wrappers and every declaration for it has to run before
  the first launch from any of them.
- **The fork axis is not only the dtype, which is the part that is not guessable.**
  `warp._src.utils.map` keys on `(is_array, type(input).__name__, dtype, ndim, broadcast_mask)` per
  input, where `broadcast_mask` is `tuple(d == 1 for d in shape)` — so a **length-1** array forks a
  module (12 ops fork on that axis *alone*, and it is the normal path for every
  reduction-into-a-scalar wrapper), as does an **`indexedarray`** from a Python-scope gather, as does
  the rank. Derive the table by instrumenting `wp.map` over a suite run and recording Warp's own key;
  check 23 fails when a module needs a table and has none, and the completeness gate is the load
  census (one `map_*` load per `(module, device, block_dim)` pair is the floor). Zero-length **CPU**
  arrays are enough to declare a CUDA overload: the module is keyed by dtype, not device.
- Never declare map kernels in `triwarp/__init__.py` (§3.1), and never via `wp.load_module` /
  `wp.force_load`, which compile eagerly.
- Still a real kernel: ops needing the thread index as *data* (`init_range`, `seed_orientation`),
  whole arrays as uniform arguments (binary-search tables), scatters, and multi-element/row-indexed
  outputs.

### 3.6 Dtype conversion at Python scope

`wp.cast(expr, TargetType)` is for **kernel / `@wp.func` scope** only — there is no `wp.cast` on
whole arrays at Python scope.

For element-wise dtype conversion of `wp.array` buffers at Python scope, allocate the destination and
call **`wp.utils.array_cast(src, dst)`** (same device, matching shape). Example: `wp.bool` →
`wp.int32` `0`/`1` flags for `wp.utils.array_scan` in `flatnonzero` — do **not** add a
`bool_to_int32` gather-style kernel.

### 3.7 Sparse: assemble from keys, never through `bsr_from_triplets`; and `nnz` is a stale capacity

**No `triwarp/` module calls `warp.sparse.bsr_from_triplets` (check 27).** It sorts every triplet
on a full-width key and allocates scratch several times the matrix -- `cotmatrix` spent 9.4 GB a
call there at `lucy` for a matrix under 1 GB, and a drained mempool makes those bytes a cost
(§13.1, §16.27). The package builds CSR two ways instead, both in `triwarp/array.py`, both with no
host readback:

- **The sparsity follows from the mesh: write keys, not triplets.** A producer writes one
  `kernels/array.csr_key(row, col, n_rows, n_cols)` per contribution (the sentinel `n_rows *
  n_cols` for a slot that holds nothing) into [`csr_key_buffers`][triwarp.array.csr_key_buffers],
  with a payload that locates the contribution's value; [`csr_from_keys`][triwarp.array.csr_from_keys]
  sorts only the bits the shape needs and returns `offsets`, `columns` and each entry's first sorted
  position; a per-row value kernel then forms each entry from its contributors in producer order
  and can write the diagonal from the row it just formed. `laplacian._mesh_operator_pattern` is
  the model, shared by `cotmatrix`, `connection_laplacian`, `graph_laplacian` and `laplacian`
  (given faces): below `_UNDIRECTED_PATTERN_FROM_FACES` one sort of six directed keys per face
  plus the diagonal slots, above it one sort of the three undirected keys per face plus a stable
  32-bit sort of the unique edges by their larger endpoint for each row's transposed half -- the
  identical matrix either way, and a forced-threshold test pins the two to each other.
- **The input is genuinely an unordered coordinate list:
  [`csr_from_triplets`][triwarp.array.csr_from_triplets]**. Unpruned it is `bsr_from_triplets`
  (summed in triplet order, out-of-range triplets dropped; bit-identical on CUDA, where Warp also
  sums in triplet order, and equal to rounding on CPU, where it does not). **Pruning differs, on
  purpose**: `prune_numerical_zeros` drops every entry whose *sum* is zero, where Warp drops
  zero-valued triplets before summing and keeps an entry whose triplets cancel. So the two differ
  only in stored zeros -- every entry Warp keeps and we drop is zero there (exactly on CUDA, within
  float32 rounding on CPU). Zero-valued triplets are still skipped before the sort, which cannot
  change a sum and makes the rule free. Used where the input is a genuinely unordered coordinate
  list: the energies, `laplacian(edges=...)`, `graph.edges_to_csr`, `index_sparse`, the Loop
  operator and the implicit smoothing system.
- **A CSR the producer can write directly is written directly** and wrapped by
  [`bsr_from_csr`][triwarp.array.bsr_from_csr] (the multigrid tentative prolongator, one entry
  per row; the empty matrix).


**Never size a buffer, slice, or launch dim off `matrix.nnz`.** After `bsr_from_triplets` the `nnz`
field still holds the *triplet count it was handed*, duplicates included — so for any
duplicate-emitting build it is an upper bound, measured at **1.85x** the true count on
`laplacian.cotmatrix`, which emits six off-diagonal triplets per face plus one diagonal slot per
vertex (it was 3.4x at the old twelve per face, §16.27). Use **`matrix.nnz_sync()`** (one host
readback) or read `offsets[nrow]`, which `energies.k_harmonic` already does.

**And `nnz` is a *cache*, not a fixed field: `nnz_sync()` repairs it in place.** Nothing else syncs
it (`bsr_mv`, `values.numpy()`, `offsets.numpy()` all leave it stale). So whether a `.nnz` read is
correct depends on whether unrelated earlier code happened to sync that matrix, which makes the bug
order-dependent and is a live trap **for the test as much as the code**: a guard that measures the
capacity and then hands the *same* matrix to the function under test has already repaired it, and
passes against the broken implementation. Build two operators — one to measure, one to hand over
(see `test_filter_laplacian_implicit_duplicate_built_operator`).

The failure is silent and it is not a Warp bug. Sizing a `triplet_buffers` allocation by `nnz` leaves
the tail `[nnz_sync(), nnz)` unwritten, and since those buffers are `wp.empty` (§3.3, deliberately)
the gap reaches the next `bsr_from_triplets` as **uninitialized triplets**. `bsr_from_triplets` drops
an out-of-range row/column index silently — verified for both a huge index and a negative one, no
exception and no CUDA fault — so most garbage vanishes and the answer looks right; the entries whose
garbage index happens to land in `[0, nrow)` accumulate a garbage value into a **real** entry.
Measured on `_build_implicit_system` with a `cotmatrix` operator, that is a result eleven orders of
magnitude off the correct one. **This is what the long-standing "`bsr_mm` is nondeterministic on
CUDA" claim really was** — `bsr_mm` is sound; do not reintroduce that explanation.

Two corollaries. A matrix built by *duplicate-free* triplets (`laplacian.laplacian`) has
`nnz == nnz_sync()`, which is why the default paths never showed this — so a probe on the default
operator proves nothing, and the check belongs on a `cotmatrix`-shaped input. And where a triplet
writer legitimately leaves slots unwritten (a conditional emit, as the region solves' writers were
before §16.19 assembled them directly), `wp.zeros` rather
than `triplet_buffers` is correct **for correctness** — but it is a **cost** defect: see §12.7, where
unwritten `(0, 0, 0.0)` triplets measured **31.8x**.

**This is a pattern rather than one API's quirk: a Warp object's `*_count` / `.nnz` field is a
*capacity* until proven otherwise.** `wp.Volume.get_voxel_count()` and `wp.volume_voxel_count` report
the grid's allocated capacity, not its active voxel count, so `triwarp.voxels` goes through
`Volume.get_active_stats().voxel_count` and says so at both sites. Before sizing a buffer, a slice or
a launch `dim` off any such field, probe it against a construction whose true count you know
(a duplicate-emitting triplet build, a sparse voxel set) and record the number where the rejection
lives. The failure mode is the same both times: the wrong reading is an *upper* bound, so nothing
raises and the tail is garbage.

More `warp.sparse` behaviour — `bsr_mm`'s structural superset, `bsr_compress`'s in-place return,
operands read from the stale `nnz` field — is in §12.7.

### 3.8 NumPy at Python scope is sanctioned; leaking it through the API is not

`warp-lang` carries an unconditional `Requires-Dist: numpy` and `import warp` loads it eagerly, and
`wp.array(list, dtype=...)` itself ends in `np.asarray`. So NumPy is present wherever triwarp runs,
it is a declared core dependency, and deleting `import numpy as np` from a wrapper shrinks nothing —
it only moves the same NumPy call into Warp, more slowly. **Do not open a "remove NumPy" pass**; 13
modules import it and that is correct.

Host-side metadata math (offset scans, launch dims, per-loop sizes, small candidate tables) and
host-*sequential* algorithms (patience sorting in `combine`, DP traceback in `holes`, `lexsort`
Delaunay in `reconstruction`, `argsort` + `searchsorted` chain linking in `intersection`, most of the
procedural mesh templates in `creation`) stay in NumPy: they are not device work, and porting them
buys Python loops.

**The exception is a template that is a closed-form parallel *map* whose output scales with a
resolution parameter** — no sequential dependence between elements, so a kernel buys no Python loop
and the host build is pure assembly plus an upload. `creation.icosphere` and `creation.grid` are both
in that class and both are kernels: `grid` was 78 % NumPy prologue and came out **33x faster and
bit-identical** (112x at the top of its resolution axis), because a `float64` kernel followed by the
same `float32` store rounds the same way the host build did. Decide by whether the elements depend on
each other, not by which module the function lives in.

**The census has been taken, and it is the reason not to re-open this.** An AST scan of
`triwarp/` (excluding `kernels/`, which has none) finds **488 host-side sites**: 329 NumPy calls,
67 `.numpy()` readbacks, 55 `math.*` and 32 `.tolist()` / `.item()`. Bucketed: 126 metadata and
marshalling, 81 fixed-size 3x3/4x4/scalar math, 67 readbacks, 55 scalar math, 36 host-sequential,
32 host converts, 15 module-scope constant tables evaluated once at import, and ~76 lattice and
template index arithmetic. **Only the last bucket contained anything convertible** (§16.4's
`parametric_surface`), and the rest are settled by one measurement rather than site by site: a
minimum device round trip (one `wp.launch(dim=1)` plus one host readback) is ~43 µs against
0.7-6 µs of host arithmetic for a 4x4 matmul, a 3x3 determinant or a 3x3 SVD — a **26-60x loss**,
and structural rather than an implementation detail. That disposes of 262 of the 488 sites (54 %) at
a stroke, and the module-scope tables cost nothing per call.

**The crossover, per operation class** (result staying *on the device*; ratio numpy/device, >1 = the
device wins). Use it to price a conversion before writing one:

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

And the *readback* question separately — mask already on the device, caller needs a host bool:
`mask.numpy().any()` against `reduce.any` crosses over at ~0.5 M elements, which reproduces §12.4's
figure exactly. `np.any` short-circuits, so its cost depends on the data and not only on `n` — an
all-False mask is the worst case and the one a convergence loop actually hits; timing a half-True
mask makes numpy look 100x better than it is.

**Measured NumPy share of real public calls**: nine of eleven probed sit at **0.6-6 %** with the
share flat or falling across a 256x face range, which §9 calls a decline. The two above 10 % were
`creation.uv_sphere` / `capsule` (which is `_revolve_kept_template`, the filter that licenses the
closed-form fast path — under the device floor for profiles up to 64 points, so a wash below ~256
profile points and declined) and `bounds.oriented_bounding_box`, where the NumPy was **not** the
cost (§16.4).

Three things are still defects:

- **A public signature or return that names `np.ndarray`**, which forces the dependency on the
  *caller*. Return `wp.mat33d` / `wp.vec3` (`measures.moments` returns the inertia tensor as
  `wp.mat33d` — `wp.mat33` would discard the `float64` digits the integrals exist to keep); annotate
  inputs `Sequence[Sequence[float]]` when the body is a duck-typed `np.asanyarray`, which is
  *widening*. The one sanctioned exception is `triwarp/io.py`, where meshio hands back `np.ndarray`
  unconditionally and NumPy-in is `mesh_from_numpy`'s entire purpose.
- **NumPy standing in for a Warp Python-scope equivalent that exists** — *but price the operation
  first, because §13.1 measured this rule running the other way for several of them.* A Warp builtin
  called at Python scope is a builtin *dispatch*, so `wp.length` is 13-14x `np.linalg.norm`,
  `wp.min`/`wp.max` on a `vec3` 37x `np.minimum`, `wp.inverse` 3.2x `np.linalg.inv` and `vec3 - vec3`
  17x, while `wp.cross` genuinely beats `np.cross` by 1.3x. What this bullet is really about — not
  forcing a host round trip, and not naming `np.ndarray` in a public signature — is unaffected.
  `wp.full`, `arr[k:].fill_()`, `wp.array([wp.mat44(...)])`, `wp.determinant`, `wp.inverse`,
  `wp.transpose` and `wp.svd3` all work at Python scope and need no host buffer; `arr.list()[0]`
  gives a row-indexable `wp.mat44` from a `wp.array[wp.mat44]`. `math.pi` / `float("nan")` /
  `float("inf")` beat `np.pi` / `np.nan` / `np.inf`. One trap: **`wp.svd3` is not a substitute for
  `np.linalg.svd` of a non-square matrix** — `creation._align_vectors` takes the SVD of a `(3, 1)`
  for basis completion and its free rotation about the axis is a *gauge* the trimesh comparison pins
  element-wise.
- **A `.tolist()` immediately splatted into a Warp vector/matrix constructor is noise — delete it.**
  `wp.vec3(*x.tolist())`, `wp.mat33(*x.ravel().tolist())`, `wp.mat44(*x.flatten().tolist())`,
  `wp.mat33d(*x.ravel().tolist())` all construct identically from the raw NumPy array with the
  `.tolist()` dropped — verified for `float32` and `float64`, for a contiguous row and for a
  non-contiguous (transposed / column / negated) view. It changes neither the dtype-narrowing a
  `wp.mat33`/`wp.vec3` float32 constructor does (a Python `float` and a `np.float64` round to the
  same bits) nor anything else observable; it only allocates a throwaway list. The same redundancy
  one level down is `math.dist(a.tolist(), b.tolist())`, where `math.dist` also takes a raw NumPy
  array directly. Two shapes are **not** this defect and must stay: a `.tolist()` whose result **is**
  the return value, satisfying a genuine `list[...]`-typed public signature rather than feeding a
  Warp constructor (which is what keeps a public return from naming `np.ndarray`, the very thing
  this section's first bullet forbids); and a `.tolist()` used to get plain Python scalars for
  non-Warp bookkeeping such as a dict key. §7.1's test-writing convention
  (`wp.vec3(*array_np.tolist())`) is unaffected and stays the sanctioned spelling in `tests/` for
  consistency — do not carry it into `triwarp/` production code as though it were load-bearing.
- **`arr.numpy().tolist()` is `arr.list()`, but only for a rank-1 array.** `wp.array.list()`'s
  scalar-dtype branch is literally `self.numpy().flatten().tolist()`, so for an already-1D array the
  two spellings return the identical Python list at identical cost — `.list()` calls `.numpy()`
  internally, so there is no speed win, only one fewer visible readback call. **It is not a
  substitute for a rank-2 (or higher) array: `.list()` unconditionally flattens**, where
  `.numpy().tolist()` preserves the row structure. An `(n, 2)` output as `.list()` returns one flat
  `2n`-element list rather than `n` pairs, which silently breaks anything iterating rows or indexing
  into it; both patterns exist in the tree and were left alone for exactly this reason. Restrict the
  swap to a genuinely `ndim == 1` buffer — an offsets/index/mask array, a `list[wp.array]` element,
  or an already-indexed row of a 2D array — which is what `array.py`'s `split` already asserts
  before this exact pattern, and is the cheap tell to check before converting a site.
- **NumPy reducing a full `.numpy()` readback** — `.min()`, `.max()`, `.any()`, `.sum(axis=0)` — is a
  §9 defect wearing NumPy's clothes: the whole array crossed the bus to produce one scalar. Use
  `triwarp.reduce` (or `wp.utils.array_sum`, which reduces a `wp.vec3d` array componentwise and so
  needs no kernel of its own), and check whether a kernel for it already exists before writing one —
  `holes._mean_rim_edge_length` was reading back the entire vertex buffer while `_loop_perimeters`,
  three hundred lines up in its own file, already computed the answer on the device. **Decide these
  on the CUDA measurement and accept the CPU regression** (§9), but keep the host path where the
  buffer never scales with the mesh — a `k`-element argument check, a per-segment offset list —
  rather than with the vertex or face count. The seven-site A/B and its four rejections are §14.5.

### 3.9 Device checks and the launch-device memory-safety rule

**Every public function that accepts two or more device-bearing arguments calls
`_device.require_same_device(**named)` as its first statement**, passing every array, mesh, BVH,
hash grid, `wp.Volume` or list/tuple of those it received — including an `X | None = None`
precomputed-cache argument; the helper skips `None` silently, so nothing is filtered beforehand.
It raises `RuntimeError` (deliberately not `ValueError` — see below) naming the two mismatched
arguments and their devices. Document it with a `Raises` entry the same way a direct `raise` is
documented (§4.3) — check 11 cannot see through the delegation, but a caller reading the docstring
should not have to know that.

This reverses an earlier revision of this file (*"do not check that input arrays share the same
device"*), and the reversal is deliberate. That rule was correct for triwarp's own internal call
sites: every kernel factory forwards `device=` from an input array (§2.1), the test harness runs
under `LaunchArrayAccessMode.STRICT`, and an internal wrapper calling another triwarp wrapper
already passes arrays it just validated or produced itself. None of that holds for an external
caller of the *public* API, who has no reason to know Warp has a device model at all and who can
trivially construct a mismatch by accident — one mesh loaded from disk (landing on CPU) and one
built with a GPU default. For that caller, the two ways a mismatch actually fails (below) are not a
`ValueError` away; they are silent memory corruption or a bare segfault with no Python traceback. A
cheap comparison of a handful of `.device` attributes, once, at the public boundary, converts an
undebuggable native failure into an ordinary Python exception, for a cost that is unmeasurable next
to a single `wp.launch` (§13.1). **The rule is still "do not scatter ad hoc checks through internal
helpers"** — it is now "the public boundary checks once, through one shared helper, and everything
behind it stays trusting."

**But do not believe that `wp.launch` raises on a device mismatch — it has not since Warp 1.14.**
That release removed the unconditional same-device check (`NVIDIA/warp` GH-1461) so that
hardware-coherent launches would be legal, and the default `wp.config.launch_array_access_mode` is
`RELAXED`, which passes the pointers straight through and validates nothing. On this box the
consequences are asymmetric and both are silent:

- **CPU arrays, CUDA launch** (a launch that forgot `device=`, resolving to `cuda:0`): the GPU reads
  the host arrays over HMM (`is_cpu_memory_access_from_gpu_supported` is `True` here) and computes
  the **right answer** — then the launch is *asynchronous*, so when those host arrays are freed while
  the kernel is still running the heap is corrupted and the process aborts in `malloc` much later.
- **CUDA arrays, CPU launch**: immediate `SIGSEGV`, no Python exception (GH-1693).

Five consequences:

- **`tests/conftest.py` sets `LaunchArrayAccessMode.STRICT`**, the only mode that rejects a *genuine*
  cross-device argument; `CHECKED` validates addressability, which HMM genuinely provides, so it
  permits the launch and still corrupts. The full suite passes under `STRICT` — keep it that way.
- **`STRICT` alone would not have caught the bug that motivated it, which is why check 15 exists.**
  It only fires when an argument is *not* on the launch device, so on a CUDA run — this box, and CI —
  an omitted `device=` resolves to `cuda:0`, which *is* the arrays' device, and nothing is rejected.
  The corruption then waits for a CPU run. `test_launches_name_their_device` scans every launch site
  statically and is the half that sees it; the two guards cover different halves.
- **§2.1's "always forward the `device`" is a memory-safety rule.**
- **`.numpy()` is not a sync on a CPU array.** On a CUDA array it synchronizes; on a host array it is
  a zero-copy view, so "I read the result and it was correct" proves nothing about whether the kernel
  finished.
- **`require_same_device` raises `RuntimeError`, matching the exception type PyTorch raises for the
  same class of mistake, deliberately with different, triwarp-specific wording** — triwarp's message
  names the caller's own keyword-argument names and both devices, and suggests the fix.
  `RuntimeError` rather than `ValueError` because this is not a bad *value* in the NumPy sense (the
  arrays are each perfectly valid on their own device) — it names the same failure class PyTorch, a
  library with the same multi-device model, already reserves `RuntimeError` for.

Full root-cause history, the free-while-in-flight measurements and the two bisection techniques that
found them: §12.1.

### 3.10 Readbacks

- **Budget the host-device syncs.** Every `.numpy()` / `int(<device value>)` readback in a wrapper
  carries a comment naming why it is unavoidable. When the caller can supply the bound the readback
  infers, expose it as a keyword (`face_adjacency(n_vertices=...)`,
  `hash_indices_rows(validate=False)`) **and pass it from every in-repo caller that knows it** — an
  escape hatch nothing uses is not an optimization.
- **A readback costs ~0.1 ms; an extra device pass costs 0.9-2.4 ms.** Trading one readback for an
  extra pass is usually a *loss* (§13.1, §14.6).
- **Several adjacent small values: `triwarp._device.read_values(arr, start, count)`**, a cached
  pageable scratch and one offset copy. A slice view plus `.numpy()` of three `int32`s measured
  26 us against 10 for `read_scalar`, which was most of a closed-mesh regression in §16.25.
- **Use `triwarp._device.read_scalar(arr, index=-1)` for a tail read**, not a hand-rolled spelling —
  the fast path is device-split and a pinned scratch is a **race** (§12.1). It takes any index and
  any dtype, so `arr[k : k + 1].numpy()[0]` and `arr.numpy()[k]` are both it, spelled slower.
- **That is a *single*-value rule, and it inverts at two.** A readback's cost is almost all fixed,
  so **one `.numpy()` of a small buffer beats two `read_scalar` calls**. So a function returning
  several small device values should write them into *one* buffer and read it once, which is what
  `points.fit_plane` / `principal_axes`, `measures.moments` and `registration.icp_point_to_plane`'s
  scalar accumulator all do (1.27-3.00x, §16.4). The rule only reverses once the buffer is large
  enough for the copy to matter, so a *tail* read of a mesh-sized array stays `read_scalar`.

---

## 4. Evolving the public API

### 4.1 Naming

- **Name a function after what it returns, in NumPy vocabulary — never after the Warp call it
  wraps.** `sort_pairs` named `warp.utils.radix_sort_pairs`'s key/value mechanism rather than its
  result (a sort *and* an argsort), which is why it became `sort_and_argsort`.
- **A mask is named `<element>_<property>_mask`, element first.** Element-first sorts and completes:
  type `tw.validation.face_` and every per-face predicate appears, which is what the property-first
  spellings did not do. But **a convention followed everywhere regardless of fit is not worth
  having**: `radius_outlier_mask` / `statistical_outlier_mask` (the property *is* the name),
  `half_space_mask` (the element is implicit and the geometry is the point), `uv_seam_vertex_mask`
  (element-final, and the qualifier is a namespace) and `convex_subset_mask` /
  `convex_superset_mask` (the module docstring turns on the subset/superset opposition) are
  deliberately left alone. Check 2 scans these names for their dtype; nothing enforces the *order*,
  which is a review question.
- **Two public names that differ by one character are a defect even when both are correct.**
  `boundary.boundary_loop` and `boundary.boundary_loops` meant "the longest one" and "all of them";
  the singular is now `longest_boundary_loop`. Look for this whenever a plural is added next to an
  existing singular.
- **A wrapper whose whole device side is one kernel nobody else launches shares that kernel's
  name.** The affix classes to reject are `init_` / `do_` / `compute_` / `run_` / `make_` /
  `kernel_` / `_impl` / `_inner`, which say nothing a reader did not already know and only stop
  `grep <name>` from finding both halves at once.
  **Do not apply this literally — it is a rule about *filler*.** An AST scan pairing each wrapper
  with the kernels it references, restricted to wrappers referencing exactly one kernel that no
  other wrapper references, flags 129 sites; narrowing to "the two names differ only by a filler
  affix or a plural" leaves **6**, and every one of those six is *informative*:
    - a **plural** because the kernel is batched over what the wrapper answers for one thing
      (`geodesic_walk.trace_from_face` → `trace_from_faces`);
    - **`_pass` / `_step`** because the kernel is one iteration of a loop the wrapper runs;
    - **`finalize_`** because the kernel is the second half of a two-stage reduction whose first
      half is another wrapper (`points.fit_line` / `fit_plane` / `principal_axes`).

  Two further exemptions the 129 make obvious. A kernel in a **shared kernel library** (§3.1's four)
  keeps that library's vocabulary — `vertices.vertex_defects` launches `scatter.scatter_sum_scalar`
  and must, because the name has to read correctly for the other importers. And a kernel that
  computes one *ingredient* of the wrapper's answer keeps the ingredient's name, the rest of the
  wrapper being host-side index arithmetic: `remesh.subdivide` → `compute_midpoints` is right about
  the midpoints and would be wrong called `subdivide`. So the question to ask is not "do the names
  match" but **"does the kernel's name carry information the wrapper's name does not"**.
- **A tuning choice is a keyword, not a name — and if the kernel already branches on it, the Python
  layer is the only place it doubled.** `neighbors` exposed each ball and nearest query twice, once
  per accelerator, for eight names covering four operations, while `kernels/neighbors.py` had
  *already* unified them behind `ACCEL_HASHGRID` / `ACCEL_BVH` selectors. They are now `query_ball` /
  `query_ball_count` / `query_ball_with_offsets` / `query_nearest` with
  `backend="hashgrid" | "bvh"` and an `accelerator=` that infers it. Three consequences:
    - **A default that must be distinguishable from "not passed" is spelled `None`.** `backend`
      defaults to `None`, documented as "`hashgrid` when no `accelerator` is given", so that handing
      over a `wp.Bvh` and nothing else does not read as contradicting a default the caller never
      wrote. Only an *explicit* mismatch raises.
    - **Keep the discriminator in the benchmark group name, not the function name** —
      `query_ball_bvh` / `query_ball_hashgrid`. The group name is the parity key, so all the markers
      move in the same commit.
    - **A merge like this needs a triwarp-against-triwarp test that the two paths agree**, or the
      shared group name is an unchecked claim (`test_the_two_backends_agree`).

  What stays split: `query_bvh_ball` and `query_bvh_box` are genuinely BVH-only — a hash
  grid has no box query. And where the *pairings are different algorithms over different inputs*
  rather than one algorithm with a tuning knob, the name keeps carrying the type: `metrics.chamfer_*`
  / `hausdorff_*` were considered for the same treatment and declined.
- **One operation family, one module.** Functions with the same shape of signature computing the same
  *kind* of answer belong together, and a family split across two modules is a defect however
  reasonable each half looked when it landed. The five per-point surface descriptors
  (`ambient_occlusion`, `volumetric_obscurance`, `shape_diameter`, `thickness`,
  `max_tangent_sphere`) sat in two modules while sharing one kernel module, and
  `benchmarks/test_proximity.py` had already voted by holding all five rows — **the suite disagreeing
  with the split is the signal to look for.**
- **Place a function by what it computes and what machinery it shares, not by where the reference
  library keeps it.** `triangles.volume` / `moments` / `centroid` sat in `triangles.py` because
  `trimesh.triangles.mass_properties` exists, and they are whole-mesh reductions in a per-triangle
  module. Names keep their trimesh / igl spelling where that is the field's vocabulary — this rule is
  about *placement* only. **The machinery half outranks the subject half, and it is measurable**: a
  function that is a *component* of another module's solver stays with it. Check with an import/call
  scan, not by reading.

### 4.2 Signatures and returns

- **A packed buffer and its offsets are returned, and accepted, values first.** `(values, offsets)`,
  never `(offsets, values)`; where a third array rides along it is a *per-item* one and goes last
  (`(ring_halfedges, offsets, is_boundary)`). The convention is stated in `array.pack_1d_arrays`'
  docstring. Both halves are usually `wp.array[wp.int32]`, so **a transposed unpack type-checks,
  runs, and indexes garbage** — there is nothing but the convention to lean on.
  `igl.vertex_triangle_adjacency` orders it the same way. The one place a caller *constructs* such a
  pair by hand is a precomputed keyword (`descend_field(vertex_faces=…)`); transposing it there
  **segfaulted the CPU backend several launches after the call, not at it**, so that composition
  carries a test of its own. Do not debug such a crash as a Warp problem — check the pair order
  first.
- **The offsets are always the `n + 1` total-terminated form, package-wide (round 23)** — scipy's
  `indptr`: item `i` is `values[offsets[i] : offsets[i + 1]]`, `offsets[-1] == len(values)`, an
  empty result is `[0]`, and there is **no separate sizes array** anywhere (a size is a difference
  of neighbours, one load either way). `array.split` enforces it and raises on a length-`n` offsets
  array, which is the ambiguity one convention removes; `counts_to_offsets` has no `include_total=`
  any more. Build one by allocating the `n + 1` buffer, writing the counts into `offsets[1:]` and
  scanning in place — that one idiom saved an allocation at seven sites. **The hazard of the form
  is an item count read as `offsets.shape[0]`**: it type-checks and runs one item past the end,
  which on the CPU device is host-heap corruption (§12.1), so a convention change is gated by
  `tests.devices`, never a CUDA run alone.
- **Every function returning a list of variable-length items has a packed sibling
  `<name>_with_offsets`**, returning device arrays (values, offsets, then any per-item array) with
  no per-item Python, and the list form is that plus `array.split` — so the two cannot drift, and
  the list costs ~0.4 µs an item (§13.1's stamped views). A function that *takes* the packed form
  is `<name>_from_offsets` (`boundary.loop_perimeters_from_offsets`); a function whose input is
  already packed and whose output is the list (`geodesic_walk.trace_polylines`) needs no sibling.
  `_batched` is not a suffix in this package. `array.split` also accepts the offsets as a host
  sequence, for a caller that already holds the bounds (both `marching_triangles` links finish
  with them on the host) — an upload and a readback fewer.
- **No public signature or return type may name `np.ndarray`**, outside `triwarp/io.py` (§3.8).
- **A guard must encode a real limitation.** When the implementation is naturally rank- or
  dtype-agnostic — a flatten/reshape, a generic `@wp.func` — drop the `ensure_ndim` cap and widen the
  annotation instead of validating a restriction that is not there.
- **A `Literal`-typed *menu* argument is validated at the public boundary and raises `ValueError`
  naming the argument, the offending value and the options. The annotation is a hint, not a
  guard.** basedpyright rejects an off-menu *literal*; nothing rejects the same value arriving
  through a variable, a config dict or a `**kwargs` splat, so a menu read as `A if x == 1 else B`
  silently answers a different question. Measured by calling every one of the **43**
  `Literal`-annotated parameters in `triwarp/*.py` with an off-menu value: **38 already raised**
  exactly that `ValueError`, and the three that did not (a nearest-neighbour fallback returning a
  plausible image, two bare `KeyError`s naming neither the argument nor the alternatives) are
  converted. Four details worth having:
    - **Two `Literal` shapes are not menus and need no guard**: the `Literal[True]` /
      `Literal[False]` pairs on a `return_*` keyword, which are `@overload` stubs over a `bool`
      where every value is legal, and a rank literal (`wp.array[DType, Literal[3]]`).
    - **Where the options live in a table, derive the message from it** —
      `f"match must be one of {list(_UV_MATCH_MODES)}, got {match!r}"` — so it cannot drift from
      the table. `list(...)` for an *ordered* table (a dict or tuple, so the message reads in the
      docstring's order) and `sorted(...)` for a `frozenset`. Where the branch is an `if` / `elif`
      chain, spell the names out, as most of the 38 do.
    - **A delegating wrapper does not repeat the check**; it documents the `ValueError` and lets
      the function it delegates to raise (§4.3, the same rule as `require_same_device`). **20 of
      the 43 are that shape** — a private validator or another public function. But the *test* then
      has to call through the wrapper: that is the only thing that shows the guard is still reached.
    - **No static check guards this, deliberately.** A permissive scan (does the body mention the
      parameter?) passes the defect that motivated the rule, and a strict one (does the body contain
      an `In` / `NotIn` test on it?) flags every delegating wrapper — **20 of 43**, nearly half the
      sites as allowlist, which §4.5 says is how a check gets switched off. The gate is instead a
      **probe**: call each menu entry point with an off-menu value and read the exception type.
- **When a function mirrors a NumPy one, mirror its *positional* signature too, and make `device`
  keyword-only.** `array.arange(n, device)` / `arange_step(count, step, device)` were two functions
  covering the one NumPy call whose whole convention is its positional arity, so a caller who knew
  `numpy.arange` had to learn a second spelling and a reader could not tell `arange_step(6, 3)`
  from `arange(6, 3)`. They are now one `arange(start, stop=None, step=1, dtype=wp.int32, *,
  device)`, which is `numpy.arange`'s signature with `device` promoted from optional to required
  (nothing here allocates onto Warp's ambient device, §3.3). Two things that fall out:
    - **A required keyword-only `device` breaks every positional call site, and the residual set
      must be re-derived from the *new* name** (§4.4). A `tw.array.arange(` grep found 15 and
      missed 4 more reached through `from triwarp.array import arange`; those failed at runtime,
      not at lint or type-check time.
    - **Keep the specialised kernel for the common case.** One `start + i * step` kernel would be
      the tidy answer and costs two more marshalled arguments on the only path any in-repo caller
      takes, so `arange` dispatches: the zero-argument `arange` kernel when `start == 0 and
      step == 1`, `arange_affine` otherwise.
- **No speculative generality.** Add an axis, parameter, or mode only when an in-repo call site needs
  it. The absence of a caller is a reason not to build it, not a gap to fill.
- **No near-duplicate wrappers.** Two public functions that are the same algorithm with different
  returns share one private helper (`concatenate` / `pack_1d_arrays` behind `_pack_segments`).
- **Inverse and dual pairs cross-reference each other and have a round-trip test** — e.g.
  `flatnonzero` / `indices_to_mask`. Bidirectional `See Also` is required for inverse pairs and for
  simple/advanced variants of one operation; it is *not* required for hub→spoke references (most of
  the ~250 one-way links in the package are correct — `cotmatrix` should not list every consumer).
- **Coverage is per module.** Every public `triwarp/<module>.py` gets both `tests/test_<module>.py`
  and `benchmarks/test_<module>.py` (check 4), and a function's tests live in the file mirroring
  *its* module (§5), not in a neighbour's.

### 4.3 Docstring, signature and body must agree

- A documented `Raises` must be reachable, and any function with a direct `raise` in its own body
  needs a `Raises` block (check 11; the functions that delegate validation to a shared guard and
  document its `Raises` are correct and are not scanned — the largest such family is every public
  function with two or more device-bearing arguments, which documents
  `_device.require_same_device`'s `RuntimeError` per §3.9).
- A documented validation must actually be performed, or the claim goes.
- Annotations must cover every rank and dtype the docstring claims and the body supports.
- **When a comment and the body disagree, decide which one is load-bearing before "fixing" it — the
  usual answer is the comment.** `tangent_space.any_perpendicular`'s comment claims it crosses with
  *"whichever coordinate axis the normal is least aligned with"* while the body compares only
  `|n[0]|` against `|n[1]|`. The body is **correct for its purpose** (it needs any axis not parallel
  to the normal, and x or y always qualifies), and "correcting" it into a three-way argmin would move
  the tangent frame at every z-dominant normal. Fix the sentence, leave the branch, and say in the
  commit which of the two you changed and why.
- **When a private helper's docstring states a rule the public surface must obey, the defect is in
  the paths that *bypass* the helper — and the `@overload` stubs and the `Returns` block are two
  further statements of that rule that no test reads.** `neighbors._shape_nearest` exists to
  *"collapse the `(m, k)` result to the rank the caller's `queries` / `k` imply"* and collapses to
  rank-1 at `k == 1`; `query_nearest`'s two degenerate early returns were taken **before** it and
  came back rank-2. The `k: Literal[1]` overload and the `Returns` prose independently declared
  rank-2 as well — one intent, four statements, one of them right. This is §2.4's duplicated
  *decision rule* one level up, and the fix is the same shape: route the bypassing paths back
  through the helper rather than repeating the collapse at each site. **After changing a
  rank/shape/dtype rule, grep for the early returns and the `@overload` stubs, not just the main
  path** — basedpyright cannot see the mismatch, because a wrong overload return type is
  self-consistent.
- **A test that hardcodes one value of the parameter the contract turns on is testing the one case
  that cannot fail.** `test_query_nearest_empty` pinned `k = 2` — the only `k` where the collapse
  above is a no-op — so the whole suite was green against it. §7.4's vacuity rule usually reads as
  "check the fixture is not degenerate"; this is the same rule about a *parameter*, and the cheap
  version is to parametrize over the boundary value rather than a comfortable one.

### 4.4 Moving or renaming

**Moving or renaming a public function moves everything derived from it — in the same commit.** A
move is not done when the wrapper compiles; it is done when nothing still points at the old home.
Five artifacts, every time:

1. **Its kernels, if they are exclusively its.** A kernel referenced by only the moved function moves
   to the destination's `kernels/` module; a kernel shared with a function that stays put does
   **not** move, and the new kernel module imports it (kernel-to-kernel imports are normal —
   `kernels/predicates.py` has many importers). Decide by measuring, not by reading: an AST scan of
   which wrappers reference each `kernel_<mod>.<name>` is the authority, because a kernel that
   *looks* single-purpose is often reached from a private helper in a third module.
2. **Its tests**, into `tests/test_<destination>.py`, keeping the §5 source order.
3. **Its benchmark rows**, into `benchmarks/test_<destination>.py`.
4. **Its `benchmark(group=...)` name**, when the group is named after the function or its old module.
   Renaming a group is allowed and sometimes required — but the group name is the parity key, so
   **every `parity` / `noparity` marker citing it must be updated in the same commit**, and
   `uv run python -m tests.parity` must show the same pair count before and after.
5. **Its docs entry** in `docs/gen_ref_pages.py` `SECTIONS`, plus every `[`name`][triwarp.old.path]`
   cross-reference — `zensical build --strict` is what finds the ones you missed.

**Renaming a *keyword argument* has its own artifact list, and a call-site scan sees none of it.**
Three sites survived a paren-aware rewrite of all 33 `neighbors` query calls, each invisible for a
different reason, and each caught by a *runtime* check rather than a static one:

1. **A `TypedDict` field feeding a `**splat`.** `registration._TargetIndex` declared `bvh: wp.Bvh`
   and the call site read `query_nearest(..., **target_index)` — the keyword's name is nowhere near
   the call. Grep the *old keyword name on its own*, not just at call sites.
2. **A rename that ran after the function rename.** A bulk pass had already merged the function
   names, so the later keyword pass no longer matched them. Sweep for "merged name still carrying
   the old keyword" as a separate, final pass — ordering two mechanical passes wrong silently skips
   their intersection.
3. **A fenced ```python docstring example.** Check 14's runtime `exec` is what found it; `ast.parse`
   and every grep were clean.

The general rule: after any mechanical rename, **re-derive the *residual* set from the new names**
rather than trusting that the pass which produced them was complete. And the full suite catches what
a targeted per-file run does not — a rename is not done until the whole suite has run.

### 4.5 The mechanical gate: `tests/api_conventions.py`

**Twenty-seven checks**, and they fail the default `pytest` run.

- **Eight scan the public surface of `triwarp/` (excluding `kernels/`)**: a summary line naming a
  reference library (1); a `*_mask` producer that does not return `wp.array[wp.bool]` (2); a module
  summary advertising Warp (3); a module without a `tests/` **and** a `benchmarks/` file named for it
  (4); a private name reached across a module boundary (5); one public name exported by two modules
  (6); a top-level `kernels/<name>.py` without its `triwarp/<name>.py` or the reverse (7); a private
  helper defined above its first caller (8).
- **Check 9 scans `kernels/`, `tests/` and `benchmarks/` as well**: a comment or docstring blaming a
  Warp version older than the installed `warp-lang`. It reads **only** the anchored spelling
  `Warp 1.17`. `_WARP_VERSION_ALLOWLIST` holds deliberate history and is where the next upgrade's
  re-verification notes go. **A stale `pytest.skip` is worse than a stale comment** — the comment
  misinforms, the skip deletes a branch, and on a CUDA box the deleted branch is the one nobody runs;
  16 skips for one long-fixed `cg` bug survived because check 9 originally scanned `triwarp/` only.
- **Check 5 has a `_`-prefixed-module escape, and reaching for it is usually the wrong of two
  answers.** The check reads only public wrapper modules and only `_`-prefixed *names*, so a shared
  wrapper-side helper with a plain name inside `triwarp/_thing.py` is invisible to it — which is what
  `_device.py` is. But a new `_*.py` holding one helper is a module created to dodge a check, and the
  check's premise is sound: a shared *operation* wants a home, not a hiding place. Worked through on
  `adjacency.resolve_face_adjacency`, which review wanted off the public surface while `curvature`
  and `validation` both called it. Moving it to `triwarp/_adjacency.py` passed every gate and was
  still reverted, because splitting it showed which half was actually shared: the **rule** (both
  tables or neither) is a contract three modules must enforce identically, and is now the public
  `adjacency.require_paired_adjacency`; the **derivation** is one `face_adjacency(return_edges=True)`
  call that each caller writes inline with its own `n_vertices`. So when a private cross-module
  helper has to stop being public, first ask whether it is really one operation — a validator plus a
  one-line default is two, and only one of them needs to be reachable. Two knock-ons either way:
  every `[`x`][triwarp.mod.x]` cross-reference to a name that moves into a `_*.py` breaks
  `zensical build --strict` (a private module generates no page), and a *newly* public validator
  needs its own `Raises` block (check 11) and a test that covers the accepting cases as well as the
  raise.
- **Check 10**: an allocation with no `device=` (§3.3). **Check 11**: a public function that raises
  with no `Raises` block (§4.3).
- **Check 12**: a fenced ```python docstring example that does not run. The one static check that is
  not static — `tests/api_conventions.py` extracts the blocks and `tests/test_api_conventions.py`
  `exec`s them against a mesh fixture, because both defects it was written for were *runtime* ones (a
  `wp.array` compared with a float, a NumPy bool array handed to `flatnonzero`) and `ast.parse` sees
  nothing wrong with either. Blocks holding a bare `...` are deliberate outlines and skip. A new
  example needing a name the fixture does not bind fails with `NameError` — extend
  `example_namespace`, do not weaken the test.
- **Checks 13-18, 20, 22 enforce kernel conventions**: `out_` prefix and position (13, §2.1);
  subscript-style array annotations, whole package (14, §1.2); a launch with no `device=` (15, §3.9);
  cast spelling (16, §1.3); integer division (17, §1.5); bare `bool`/`int`/`float` annotations (18,
  §1.2); kernel-scope ternary (20, §1.5); bare single-index `wp.tid()` (22, §1.3).
- **Check 19** reads `tests/`: a test comparing against a reference library with no class label
  (§7.4). It exists because the convention decayed **twice** — the lowercase `class b` spelling went
  21 → 0 → 9, invisible to the prescribed grep because a human reads `class B` and `Class B` the
  same. Four decisions keep it from misfiring: it keys on `ast.Assert`, not the function body (a
  fixture unpack `mesh_tm, mesh_wp = icosphere` names a `_tm` variable in every mesh test, and keying
  on the body takes it from 0 hits to 120); it accepts **all four** label phrases (§7.4), or it would
  fail 14 correct tests; it leaves `_np` out of its suffix list (290 false positives); and it checks
  only that a label is *present*, never that it is the right one. The rarer defect it also closes is a
  comparison with **no docstring at all**, which ruff cannot see because `D103` is ignored.
- **Check 21**: a MeshLib **or promesh** name anywhere under `triwarp/` — a *licensing* guard
  (§7.6). Two libraries, one scan, opposite reasons: MeshLib's licence is readable and
  restricts *use*; promesh's mirror carries no licence file at all, so a "port of" comment
  cites terms nobody here has checked.
- **Check 23**: a kernel module whose `@wp.func` is `wp.map`'d from several sites with no
  declaration table (§3.5). Like its `wp.overload` sibling it asserts a table *exists* and never that
  it is complete; the completeness gate is the load census, which is a clock measurement (§15.1).
- **Check 24**: a `.claude/CLAUDE.md` cross-reference naming a section that does not exist, **or a
  bare chapter number where that chapter is subdivided**. The second clause is the one that earns
  the check, and it is a *staleness* rule rather than a convention one: 111 references across 46
  files were citing Part I's old numbering, and every one of them still resolved. `section 4` stood
  simultaneously for §3.5, §2.5, §2.7 and §3.7; `section 6` for four different subsections of §7.
  One number meaning several sections is exactly what resolution alone cannot see. Chapters 5, 6, 8,
  9, 10 and 11 carry no `###` heading, so a bare number is their only citation and is accepted — the
  check reads that from the file's own headings rather than from a list, which is what keeps
  `_CLAUDE_CHAPTER_ALLOWLIST` empty (rejecting *every* bare chapter reports 68 sites of which 38 are
  correct citations — more than half the hits as allowlist, which is how a check gets switched off).
  It scans `triwarp/`, `tests/` and `benchmarks/`, matches `AGENTS.md` too (a symlink to this file),
  and abstains when `.claude/CLAUDE.md` is absent. **Two things it cannot see**: a reference naming
  the file in one sentence and the number in the next, and a bare `section N` belonging to a *paper*
  — `kernels/remesh.py` cites "Liepa 2003, section 3", and widening the pattern to catch the first
  misfires on the second.
- **Check 25**: a `!!!` admonition inside a numpydoc **item-list** section — `Parameters`,
  `Returns`, `Yields`, `Receives`, `Raises`, `Warns`, `Attributes` or `See Also` (§6). griffe reads
  each entry's first line as a *name*, so an `!!! note "..."` header between two `Raises` entries
  becomes an exception type: confirmed by loading such a function and getting **three** raises
  entries, the middle one carrying the literal string where an annotation belongs. The published
  page grows a row for a type that does not exist, the admonition's body becomes that row's
  description, and the warning never renders as a warning. Nothing else sees it — `zensical build
  --strict` stays clean, because every cross-reference in the swallowed text still resolves, and
  ruff's `D` rules do not model section contents. It had decayed to **eight** sites in five modules,
  seven of them in a `Raises` block, and the clustering is the lesson: an author writes the caveat
  where the thought occurs, and the thought occurs while documenting what the function rejects. **It
  ships with no allowlist and no allowlist machinery**, because there is no legitimate instance —
  every admonition has a correct home in `Notes` or in the leading description.
- **Check 26**: a Warp-typed module constant used at Python scope as an **arithmetic operand or a
  slice bound**. `wp.int32(0)` is a `warp._src.types.int32`, not a Python `int`, and its `+ - *`
  route through Warp's builtin dispatch — two to three orders of magnitude a Python float's operator,
  and worse for a `wp.array` slice, since `__getitem__` forms `stop - start` and `strides * start`
  itself (§13.1). It is the sixth member of the checks 16/17/18/20/22 family: the spelling is legal,
  the answer is right, and nothing but a scan sees it. **Scope is deliberately narrow on three
  axes**, each of which keeps it from becoming an allowlist. It reads uses only in the wrapper layer
  (`triwarp/*.py`, which holds no kernel bodies) and at *module scope* in `kernels/`, never a kernel
  or `@wp.func` body, which is where these constants belong. It keys on *constants*, not on
  `wp.length` / `wp.cross` calls, where §13.1 shows most hits would be legitimate. And it looks
  *through* `wp.constant`, because `wp.constant(7)` is a plain `int` (§12.6) and only
  `wp.constant(wp.int32(7))` is Warp-typed. It ships with an **empty** allowlist; the two correct
  spellings need no exemption, since a constant forwarded to `wp.launch(inputs=[...])` is neither a
  `BinOp` nor a `Slice`, and `int(CONST)` is an `ast.Call` operand. The wider class — vector
  arithmetic, and explicit Python-scope builtins — is not statically decidable and is covered by the
  runtime census in §15.11 instead.
- **Check 27**: a call to `warp.sparse.bsr_from_triplets` / `bsr_set_from_triplets` anywhere under
  `triwarp/` (§3.7). Empty allowlist; tests may still call it, where it builds an *input*
  independently of the code under test.

Each check carries a written allowlist — read the reason before adding an entry, and prefer fixing
the code. **The gate does not replace review**: it cannot tell whether a *new* name is a good one,
only that it does not break a convention the package already holds to.

**Two rules about the gate itself:**

- **A new Warp construct can silently switch off a static check that predates it.** Bundling buffers
  into a `@wp.struct` removed them from check 13's view entirely, because the check resolved store
  targets to a bare `ast.Name` — `edges.count[slot] = 1` and `wp.atomic_add(edges.count, slot, 1)`
  both stopped matching, and nothing went red. Fixed by walking `ast.Attribute` as well as
  `ast.Subscript`. **After introducing a construct the package has not used before** (the first
  `@wp.struct`, the first `wp.ref`, the first tile intrinsic), grep `tests/api_conventions.py` for
  checks that pattern-match on AST node types and ask whether the new construct is invisible to them.
  A green suite after a refactor is equally consistent with "still covered" and "no longer looked at".
  **A plain `@wp.func` extraction does it too**, which is the cheaper and likelier version of the
  same failure: check 13 resolves store targets syntactically, so moving a kernel's writes into a
  helper it passes the buffer to removes that buffer from the check's view entirely — caught only by
  the staleness half reporting the allowlist entry as matching nothing. **So §2.4's "extract the
  shared run" and this check are in tension, and the extraction wins**: the entry comes out and the
  convention goes on binding the parameter unenforced.
- **Keep writing checks with a staleness half.** The tell that caught the struct case was check 13
  complaining that two allowlist entries no longer matched anything.

---

## 5. Function ordering within a module

`mkdocstrings` is configured with `members_order: source`, so **source order is the rendered docs
order** — placement in the file is part of the public API's discoverability, not cosmetic.

### Python wrapper modules (`triwarp/*.py`)

- File layout: module docstring → imports → module constants / type aliases → functions.
- **Group public functions thematically, then order the groups by importance and expected frequency
  of use.** Primary entry points at the top; niche or low-level variants last. Within a group put
  closely related functions consecutively (`expand_vertex_mask` / `shrink_vertex_mask`), and the
  simple form before its advanced variants (`query_ball` before `query_ball_with_offsets` /
  `query_ball_count`).
- **Stepdown rule for private helpers** (check 8): a private function called by exactly one public
  function goes **immediately after** that function. A private helper shared by several functions in
  the same file goes after its **last** caller. Cross-cutting utilities go in a trailing "private
  helpers" section at the bottom. A private helper must never appear above its first caller — a
  reader should never need to jump backward to a definition they haven't been introduced to yet.
  Two things the draining sweep learned: reordering has to consider *every* private helper in a
  module, not only the flagged ones (moving a flagged helper below a callee it uses strands that
  callee); and a name can appear several times at module scope, since `@overload` stubs precede their
  implementation, so verifying a "pure move" by comparing functions keyed on name silently collapses
  those duplicates — compare the multiset.

### Kernel modules (`triwarp/kernels/*.py`)

The stepdown rule **inverts**: Warp resolves `@wp.func` references at kernel-decoration time, so a
`@wp.func` **must textually precede** every kernel (or other `@wp.func`) that calls it. Order kernels
to mirror their wrapper module's public-function order, and place each kernel's `@wp.func` helpers
immediately **before** that kernel; a `@wp.func` shared by several kernels goes before its **first**
user.

### Tests (`tests/test_<module>.py`)

Mirror the corresponding wrapper module's public-function order, so a reader scanning tests
top-to-bottom sees the same story as the API page.

---

## 6. Documentation (Zensical + mkdocstrings)

Docs are built with **Zensical** + **mkdocstrings** (`python` handler, `docstring_style: numpy`).
Zensical is the Material for MkDocs team's ground-up replacement for the MkDocs stack, adopted
because MkDocs has been unmaintained since 2024-08; it reads **`mkdocs.yml`** unchanged, so the
config filename, `theme: name: material`, the palette and `extra.css` are all as they were.

**Two commands, and the first is not optional:**

```bash
uv run python docs/gen_ref_pages.py     # ALWAYS first -- materializes docs/api/ + docs/SUMMARY.md
uv run zensical build --strict          # validate: exits 1 on a broken cross-reference
uv run zensical serve                   # preview locally
```

`docs/gen_ref_pages.py` generates one API reference page per public module under `triwarp/`
(`triwarp/kernels/` is excluded), so a new module needs **no manual nav entry** — but it is a
**standalone pre-build script, not a plugin hook**, because Zensical has no `gen-files` equivalent
(`zensical/zensical#51`, still open). Five facts about that runtime, each of which fails quietly:

- **Zensical ignores an unsupported plugin entry silently — no warning, no error.** A build with
  `gen-files` still listed in `plugins:` exits 0 and publishes a site with **no API reference at
  all**. `--strict` is the only thing that reports it, as one unresolved cross-reference per API
  symbol. Never register `gen-files`; always run the script first.
- **`--strict` is the cross-reference backstop** §4.4's five-artifact discipline leans on, and it
  works: it exits 1 on a dangling `[`name`][triwarp.old.path]`. The four external inventories still
  resolve under it.
- **`zensical serve` watches `docs/` but not `triwarp/*.py`.** A docstring edit does not trigger a
  rebuild, and forcing one by touching a `docs/` file does not help either — mkdocstrings has the
  module cached in-process. **Restart `serve` to see a docstring change.**
- **There is no `exclude_docs:` equivalent.** The `assets/benchmarks/*.md` sidecar tables are kept
  out of the search index by `search: exclude: true` front matter instead, emitted by
  `benchmarks/plot_comparison.py`. (`draft: true` was probed and is a no-op.) They are still
  *built*, as unlinked pages nothing references.
- **A build that finishes in ~0.1 s and leaves `site/` empty is inotify exhaustion, and it exits
  0.** Zensical adds an inotify watch for every file it reads, even under `build`, and silently
  drops every file whose `inotify_add_watch` fails -- with `ENOSPC` once the per-user limit
  (`fs.inotify.max_user_watches`, 65 536 here) is used up, which on this box an IDE's file
  watchers do by themselves. `--strict` still says `No issues found` over a site with no pages, so
  the gate reads green while checking nothing. Toy projects and the latest Zensical fail the same
  way. Confirm with `strace -f -e inotify_add_watch zensical build`. The fix is raising the limit
  (`sudo sysctl fs.inotify.max_user_watches=524288`) or closing watchers; without root, an
  `LD_PRELOAD` shim whose `inotify_add_watch` returns a fake descriptor on `ENOSPC` restores a
  full build, since a one-shot build never needs the events (the shim's source is in §16.15).
  **And pass `--clean` after a docstring change**: `build` reuses `.cache/` and can re-emit a page
  from the previous docstring in under a second, which `--strict` also passes.
- **`literate-nav` and `section-index` are implemented natively**, so neither package is installed
  — the built site is byte-identical without them — and their `plugins:` entries are read as
  configuration rather than as a request to load a plugin. `mkdocs-gen-files` *is* installed, for
  its `Nav` helper alone.

**The one-line summary says what the function returns, never which C++ call it wraps.** mkdocstrings
renders that first line as the function's entry in its module's API index, so a reference library's
name there turns the index into a table of bindings — `laplacian.cotmatrix` read *"Cotangent
stiffness matrix / discrete Laplacian (``igl::cotmatrix``)"* where it should read *"Cotangent
stiffness matrix of the mesh: the discrete Laplace-Beltrami operator."* Attribution is *wanted* and
stays — most wrapper modules mention a reference library somewhere in their prose — but one line
down, in `Notes` or `See Also`. Enforced by check 1, whose allowlist is `mesh.py`'s "mirrors
`trimesh.Trimesh`" alone. Check 3 forbids a module summary ending in `(Warp)` or `on NVIDIA Warp`:
the whole package is Warp.

**No measured timing belongs in a public function's docstring.** Not a millisecond figure, not a
speedup ratio, not a launch or byte count — those are facts about *this* box, *this* Warp version
and *this* mesh, and mkdocstrings publishes them as though they were part of the contract, where a
reader on other hardware reads a number that is simply false for them. The docstring keeps the
*claim* the number supports, in terms a caller can act on: "roughly doubles the call", "a host
readback serialises the device pipeline", "the fixed per-segment cost dominates at these widths".

**And no development-history narrative belongs in `triwarp/*.py` either — in a docstring or in a
code comment.** A sentence that reads as a lab notebook rather than documentation ("measured X,
declined Y", "reverted", "round N", "probe"/"sweep" used as a methodology rather than an algorithm
term, a cross-reference to an internal doc section) describes how the code came to be, not what it
does or how to use it. The public wrapper layer — every module under `triwarp/`, excluding
`kernels/` — is documentation for a caller, not an engineering log; keep the behavioral or
correctness fact a piece of history was attached to (a convention, a sign rule, an aliasing warning,
what raises and when), and cut the narrative around it.

**Delete it, don't relocate it.** This reverses the file's own earlier guidance, which said to move
a pruned number into a `#` comment in the same function's body. §9's "a measured decline is a
result, written at the site" still holds, but the site for that discipline is `kernels/`,
`benchmarks/` and `tests/`, not the public wrapper: a decline's reasoning belongs in the private
helper that actually pays it, or in Part II here, not in a comment a caller has to scroll past.
This is a zero-tolerance completeness rule rather than a tracked backlog: a fresh hit in a new or
edited public function is a regression to fix in the commit that introduced it.

**And the same discipline about *absolute* numbers applies everywhere, not only in `triwarp/`.**
`kernels/`, `benchmarks/` and `tests/` may carry a measured *conclusion* — a ratio, a crossover, a
share, "flat across a 256x range" — because that is what a later reader decides on. They should not
carry a wall-clock figure tied to one box and one mesh (`0.287 ms on a 40 962-vertex mesh`,
`674 ms on bunny`, a table of per-property milliseconds): it dates immediately, it is false for
anyone else, and the ratio it was quoted to support says the same thing without either problem.
Part II is the one place a raw cost model belongs, because it names the hardware once at the top.
Likewise, prose that narrates *how a change came about* — "an earlier version of this docstring",
"round N", "this used to be" where the history explains nothing a reader must act on — is
development log rather than documentation; a regression test's "X used to be a bug, this pins it"
is the legitimate exception, because that is the test's purpose.

The scan is an `ast` walk over `triwarp/`'s public functions matching
`\d[\d.,]*\s*(ms|us|µs|ns|GB|MB|kB)\b` or `\b\d+(\.\d+)?x\b` against each docstring — a plain
grep for `ms` is unusable, and the same regex over *prose* words (`measured`, `faster`,
`benchmark`) returns kilobytes of legitimate behavioural text, so key on the *quantity*. It has no
comment-scanning counterpart; a code-comment narrative slipping back in is caught by review.

Docstrings stay **NumPy-style** (`Parameters`/`Returns`/`Raises`/`See Also`), but cross-references use
**mkdocs-autorefs** link syntax, not Sphinx roles — Sphinx interpreted-text roles (`:func:`, `:attr:`,
`:meth:`, `:class:`, `:data:`, `:mod:`) have no Markdown equivalent and render as literal, broken text
in Markdown.

| Target | Syntax | Example |
|---|---|---|
| Internal (`triwarp.*`) | `` [`short_name`][fully.qualified.path] `` | `` [`face_adjacency`][triwarp.graph.face_adjacency] `` |
| External **with** a configured inventory (`trimesh`, `numpy`, `scipy`, stdlib) | `` [`fully.qualified.name`][] `` (empty brackets) | `` [`trimesh.grouping.group_rows`][] `` |
| External **without** an inventory (`warp`, `igl`) | plain double-backtick code span, no link | ``` ``warp.sparse.BsrMatrix`` ``` |
| Shapes, literals, C++ names, file paths | plain code span | ``` ``(n_vertices,)`` ``` |

- Always resolve internal refs to the **fully-qualified path**, even for a function in the same module
  — numpydoc's old auto-linking of bare `See Also` names does not carry over.
- Before adding a new external inventory to `mkdocs.yml`, verify it serves an `objects.inv`
  (`curl -I <url>/objects.inv`).
- RST admonitions (`.. note::`) don't exist in Markdown — use the `!!! note` admonition syntax.
- **An admonition goes in free prose — the leading description, `Notes` or `Examples` — never
  inside `Parameters` / `Returns` / `Raises` / `See Also` or any other item-list section**
  (check 25, §4.5). griffe reads each entry's first line as a name, so an admonition header
  between two `Raises` entries renders as an *exception type* and its body as that type's
  description. `zensical build --strict` cannot see it.
- Module-level constants / type aliases without their own docstring are only linkable because
  `show_if_no_docstring: true` is set; don't remove that option without re-checking
  `triwarp/constants.py` and `triwarp/typing.py` cross-refs.
- Every public function should have a docstring — an undocumented one still gets a page entry and
  renders with an empty description, which looks broken.
- After editing docstrings, sanity-check with `grep -rnE ':(func|attr|meth|class|data|mod):\`'
  triwarp/` — it should return nothing (`kernels/` counts too: nothing there renders, but a reader
  meets the broken role text all the same).

---

## 7. Testing

Every new geometry function MUST have regression tests comparing against a CPU reference
implementation. `trimesh` is the default; §7.6 lists the eight others and what each one actually
binds.

### 7.1 Conventions

- Test file `tests/test_<module>.py`; import pattern:
  ```python
  import trimesh.<module> as tm
  import triwarp.<module> as tw
  ```
- Use the `device` fixture from `tests/conftest.py`. **Every test function must accept `device`.**
- Generate reproducible random data with `np.random.default_rng(seed)` (a fixed integer seed per
  test).
- **Upload a NumPy mesh with `conversions.numpy_to_warp(vertices_np, faces_np, device)`**, never a
  local helper. This section used to *print the body* of one, and the result was six private copies
  across six modules at 54 call sites, differing only in where the `float32` cast sat — printing an
  implementation is an invitation to paste it. Its `wp.vec2` sibling is `numpy_to_warp_uv`; the
  inverse is `warp_to_trimesh`. Triangle-soup arrays `(n, 3, 3)` become an indexed mesh first:
  ```python
  vertices_wp, faces_wp = numpy_to_warp(
      tri_np.reshape(-1, 3), np.arange(tri_np.shape[0] * 3, dtype=np.int32), device
  )
  ```
- Call `.numpy()` on Warp outputs **inline**, before passing to NumPy comparison functions; do not
  bind a new variable for it.
- `np.allclose(got, exp, rtol=1e-5, atol=1e-5)` for floats; `np.array_equal` for booleans/integers.
- Name variables with a library suffix — `_np` (NumPy/SciPy), `_tm` (trimesh), `_wp` (Warp), `_igl`,
  `_pp` (potpourri3d), `_pml` (pymeshlab), `_o3d` (open3d), `_pv` (pyvista), `_ml` (meshlib),
  `_pmf` (pymeshfix), `_p3d` (pytorch3d). Avoid `got` / `exp`; use clear names.
- Passing a NumPy 1D vector to a `wp.vec3` scalar argument at Python scope:
  `wp.vec3(*array_np.tolist())` — not `wp.vec3(*map(float, np.asanyarray(...).reshape(3)))`.

### 7.2 Devices, and the two-process runner

The `device` fixture **is parametrized, and `--device={auto,cpu,cuda,both}` selects for *this
process*** — `auto` picks one device (cuda if available), exactly like `benchmarks/conftest.py`, and
every test id carries a `[cpu]` / `[cuda0]` suffix. `both` means "every device this process can see,
skip nothing".

**Both-device coverage is a two-process job: `uv run python -m tests.devices`.** It runs a CUDA pass,
then a CPU pass with **`CUDA_VISIBLE_DEVICES=""`**, and that variable is the whole point: **Warp's
CPU work is ~36x slower once CUDA has been initialised in the process** (§12.1), which at the pytest
level is an order of magnitude on a CPU-heavy file and the same on the whole suite. **Never reach
for `--device=both` on a GPU box to get CPU coverage** — use the runner.

**While developing, run `--device=cuda` only.** The CPU pass is roughly three times the wall clock
of the CUDA one on the whole suite and **CI runs it on every push**, so a CPU regression is caught
there; paying for it locally after each edit buys a slower loop and nothing else. Iterate on
`uv run python -m pytest tests/test_<module>.py -q --device=cuda`, and let the runner below be
something you reach for deliberately rather than reflexively.

**Reach for `tests.devices` when the change is device-dependent by construction**, not on every
edit: a `launch_tiled` kernel (§12.2), a `_device.prefers_tiled_reduction` branch, a device-gated
constant (§13.3), anything that takes `device=` from a dependency, or a kernel whose lanes
cooperate. Both-device coverage is what caught the `warp.fem` ambient-device leak in
`reconstruction._screened_poisson_adaptive` — broken for CPU input on any box with a GPU, and
invisible to a CUDA-only run (the devices matched) *and* to a `CUDA_VISIBLE_DEVICES=""` run
(`warp.fem` then defaults to CPU too). That defect class needs CUDA present *and* the arrays on the
host, which is a configuration neither single-device run reaches. `wp.ScopedDevice(device)` is the
fix when a dependency picks the device for us.

**The CPU device does still earn a local run for one thing: it is the deterministic oracle.** Float
atomics serialize there, so a byte-for-byte A/B against a baseline compares cleanly on CPU and is
noise on CUDA wherever a reduction order can move (§16.12). That is a *measurement* use, on the one
comparison that needs it, not a gate to re-run after every edit.

**A test that costs more than ~15 s on CPU wears `@pytest.mark.slow_cpu(<measured seconds>)`**, which
skips it when it would run on `cpu` unless `--device=both`. Four `screened_poisson` tests carry it
and were most of a CPU-only run — one depth-6 solve each, under a second on CUDA (§16.3). In a
CUDA-hidden process `--device=both` therefore means "all of CPU, including these", which is how the
runner's `--slow-cpu` asks for a full CPU pass. Use the marker only where the *device* is the cost
and the claim is device-independent, and put the measured number in it; a test slow on both devices
belongs on a smaller input instead.

**When a kernel is launched with `launch_tiled`, pin `"cpu"` explicitly in a parametrize** rather
than trusting the `device` fixture — it returns `cuda:0` whenever CUDA is available, so the CPU path
of every tiled kernel is otherwise unexercised (§12.2 records two defects that hid there).

**When two devices differ but neither is wrong, compare both to a common oracle rather than to each
other.** `heat_signed_distance` differs cross-device, but against `heat_geodesic` from the same loop
(which agrees across devices to 1.1e-08) both sit at the same max error — the cross-device gap is 5x
*smaller* than either device's own discretization error, so no guard is warranted.

### 7.3 Mesh fixtures (prefer over inline construction)

Reuse shared fixtures from `tests/conftest.py` instead of building meshes in each test. Fixtures
return `(mesh_tm: tm.Trimesh, mesh_wp: wp.Mesh)` via `tests.conversions.trimesh_to_warp`.

| Fixture | Use when |
|---------|----------|
| `icosahedron` | Default watertight solid, 12 vertices; inside/outside, surface sampling, sign tests |
| `icosphere`, `icosphere_coarse` | Closed and *curved* — `subdivisions=3` (642 vertices) and `2` (162). Reach for these wherever `icosahedron` is too coarse, rather than calling `tm.creation.icosphere` |
| `unit_box` | Sharp features: 12 creases at exactly 90° with 6 flat face diagonals, untranslated so a coordinate sign picks out one face. Crease / seam / dihedral-angle tests — `icosahedron` has no right angles and `cave_cube` is non-convex |
| `cave_cube` | Hollow / non-convex shell (boolean difference) |
| `hemisphere`, `half_torus` | Curved or open surfaces |
| `boy_surface` | Closed, watertight and **non-orientable**, χ = 1 — the `False` branch of `is_orientable` / `face_orientation_bits`, and `make_winding_consistent`'s impossible one |
| `mobius` | Non-orientable *with* a boundary: the same three predicates, one loop of 78 edges, χ = 0 |
| `bohemian_dome` | Closed genus 1 that self-intersects — `homology_generators` at genus 1, and `is_self_intersecting` on a *closed* input |

The last three are built by `creation.parametric_surface` and are the only inputs in the suite that
are non-orientable or that have an odd Euler characteristic. A boolean predicate asserted only on the
orientable fixtures is testing one branch; that is what these close. `creation.parametric_surface`
builds thirteen more surfaces that are not fixtures yet — reach for one (and add the fixture) rather
than hand-rolling a degenerate mesh.

- **Do not** call `tm.creation.box()` or hand-roll `wp.Mesh(...)` in tests unless the case requires a
  bespoke degenerate mesh (empty faces, unreferenced vertices). **Never construct a `wp.Mesh` with
  zero triangles on a CUDA device** — it silently corrupts the allocator (§12.1).
- When a simple cube would suffice, prefer **`icosahedron`** or **`cave_cube`** for richer geometry.
- Parametrize over multiple fixtures with `request.getfixturevalue(mesh_name)` when coverage should
  span mesh types.
- Use `mesh_wp.device` (not the `device` fixture) for query-point allocation when a mesh fixture is
  already in scope.
- Edge-case tests (empty points/faces, single-triangle pathology) may still use minimal inline
  buffers — a **single-triangle** mesh is the safe way to reach an "empty mesh" guard on CUDA.
- **The fixture *sets* are shared too**: `CLOSED_MESHES`, `OPEN_MESHES` and
  `MESHES = CLOSED_MESHES + OPEN_MESHES` live in `tests/conftest.py`. "The four that span closed/open
  and convex/non-convex" is a decision about coverage, and it had been restated in ten files under
  three names, which meant a fixture added to the set reached exactly one of them. Import them; keep
  a local list only where it is genuinely a different set, and say in a comment why
  (`test_adjacency.py` drops `cave_cube` because its coplanar box faces make every adjacency angle 0
  or pi/2).
- **`trimesh.slice_plane`'s output is a poor input to several references.** A hemisphere built that
  way from `icosphere(2)` reports **17** boundary loops in MeshLib and gains vertices in pymeshfix
  where the surface has one rim; after `merge_vertices()` it loads unchanged and reports **1**. Use
  the `tests/conftest.py` fixtures (`hemisphere` calls `merge_vertices()` for exactly this reason)
  and **assert the hole count** before comparing a per-hole answer. The same duplication once made
  `boundary_loops`' pinch handling ("last write wins") device-dependent, which read as total
  harmonic/tutte/arap disagreement; pinched rims are now walked by halfedge sector (§16.14).

### 7.4 The parity gate: a benchmarked reference must be a tested reference

`benchmarks/` asserts only shapes and finiteness, so on its own it cannot tell whether two timed
implementations compute the same thing. `tests/test_parity.py` closes that loop and **fails the
default `pytest` run** when a benchmarked `(group, library)` pair is neither tested nor exempted. Two
markers join the suites; the benchmark's `benchmark(group=...)` name is the key, which makes group
names a cross-suite API — renaming one breaks every `parity` marker that cites it.

```python
# tests/test_edges.py -- "this test proves triwarp agrees with trimesh for that group"
@pytest.mark.parity("faces_to_edges", "trimesh")

# benchmarks/test_curvature.py -- "timed, but the results are not comparable"
@pytest.mark.noparity("pymeshlab", oracle="trimesh", reason="MeshLab computes the Meyer/Desbrun "
                      "pointwise 1-ring operator, not the Cohen-Steiner/Morvan ball measure, so "
                      "its absolute value is not comparable; measured 0.982 correlation with a 7% "
                      "offset. trimesh is the oracle for this group.")
```

Both are stackable and take string **literals** only — a computed argument is invisible to a static
scan, so the scanner rejects it. Run `uv run python -m tests.parity` for the full matrix.

**Where a reference library computes the same quantity, one test must compare the two outputs.** That
is the obligation, and it is not discharged by an invariant: a function can be watertight, symmetric,
idempotent and manifold while computing the wrong answer. So if any of the nine references has the
quantity — **check, do not assume**: §7.6's hazard blocks list what each one actually binds, and
several names that *look* present are not — there is a class A/B/C comparison against it.

**Invariant checks are welcome and belong in the same test.** Watertightness, an involution, a
counting identity, a round trip, a conservation law — these catch failure modes no reference
comparison sees. Assert them *alongside* the output comparison rather than in a test of their own, so
one test carries the whole claim and the parity marker sits on the test that does the comparing.
Split them out only when the invariant needs an input the comparison cannot use.

**Where no reference computes the quantity, an invariant-only test is the honest answer**, and its
docstring says so in those words: *"Not a library comparison: <why none exists>"*, followed by what
the invariant excludes. That is a fifth label beside A-D, not a class-D exemption — D is for a
*benchmarked* pair whose results are genuinely incomparable and needs a `noparity` entry; this is for
a quantity with no counterpart to benchmark. `halfedge_twins` (no reference has a halfedge
structure), `homology.homology_generators`' tree-cotree counting identity (nothing computes a
basis) and `geodesic_walk`'s arc-length checks are the shape of it.

**Classify every comparison, and say which class it is in the docstring:**

- **A** — direct `np.allclose` / `np.array_equal`. The default.
- **B** — equal after a *named* transform, still at `1e-5`: a dict index, a unit fix
  (`igl.doublearea / 2`), a reduction (igl's per-vertex mask → triwarp's bool), a projection
  (`igl.boundary_loop` is the longest loop), `lexsort` for unordered rows, a sign or gauge fix.
  **Most apparent non-equivalence lands here.** The benchmark docstrings' "does strictly less",
  "upper bound" and "output shape differs" caveats are about *cost*, not about the value.
- **C** — a derived scalar, set distance or statistic, because no correspondence exists. Must name
  the bug class it excludes, and record a mutation probe *and its margin* in the docstring —
  threshold ≥ 3x from the measured agreement. `fraction_within` bounds must be shown to fail under
  shuffling one side, or they are testing marginal distributions rather than the correspondence.
- **D** — exemption. Only for: not an independent implementation (`oracle=` required); a different
  algorithm with a measured disagreement; a parameter the reference lacks; an answer not observable
  in isolation; stochastic with no invariant; or input classes where triwarp is undefined. **Not**
  admissible: "awkward", "the tolerance would be loose", or any class-B situation.

**Write the label as `Class A`** (capital, the word before the letter). A lowercase `class b` reads
the same to a human and is invisible to a grep, which is why check 19 gates it.

**Four phrases carry a label, not two**, and a scan or review that knows only the first two misreads
14 correct tests as unlabelled: `Class [ABCD]`, `Not a library comparison`, `Triwarp against
triwarp`, `Not a parity assert`. The last two are labels in good standing — do not reword them to
fit a narrower grep.

**Never a parity assert:** shape-only or `isfinite`-only (that is the *benchmark's* assert, and this
gate exists to stop it migrating inward); triwarp compared with itself; a threshold a constant output
would pass. A boolean assert must be parametrized over inputs producing both answers.

**Triwarp-against-triwarp is not a parity assert but is still a legitimate test**, for one job:
pinning two entry points to each other where only one has an oracle — a mask form against an index
form, a precomputed path against the deriving one, a CPU run against a CUDA one. Say which of the two
carries the oracle.

**Check the comparison is not vacuous on its fixture**, which the gate cannot do for you.

- **An empty answer.** `test_ears` compared `igl.ears` against `boundary.ears` on fixtures where
  neither library finds a single ear, so the assert was `[] == []` and the loop body checking the
  corner convention never ran. Making it non-vacuous immediately surfaced a real disagreement
  (`triwarp_opp == (igl_opp + 1) % 3`).
- **A constant answer is as vacuous as an empty one**, and in both measured cases the docstring
  *asserted the non-vacuity that was absent* — and the sentence is what stopped anyone re-checking.
  `test_connected_component_labels_random` said "200 random edges over 64 nodes gives several
  components rather than one" and produced **1** component holding all 64 nodes, so the class-B
  label-packing transform it exists for was the identity. `test_discrete_mean_curvature` said "over
  every vertex" and ran on a *regular* icosahedron, where trimesh's answer is **one value at spread
  0.0**, so a permuted result, an off-by-one in the gather and a query/vertex index swap all pass.

  **A claim about the input's shape is a claim an assert can carry cheaply, so make it an assert and
  not a sentence** — `assert np.unique(labels_np).shape[0] > 1`, `assert np.ptp(reference) > 1e-3`.
- **Assert the reference produced a non-empty answer, or its expected count, before comparing to
  it** — and treat "this function is already the oracle in tests/" as no evidence that the comparison
  is live.
- **A docstring that says the vacuous branch is "covered separately" is a claim to check, not a
  reason to stop.** `test_face_nondegenerate_mask` compared against `trimesh.triangles.nondegenerate`
  on a fixture where both sides are all-`True`, and said so — *"which is why the degenerate branch is
  covered separately by the zero-area tests in this file"*. **Those tests did not exist.** It is now
  parametrized over a clean and a degenerate arm with the expected count carrying the claim, and the
  mutation probe confirms it is the *trimesh comparison* that fails when the mask is forced
  all-`True`. **Grep for the tests a docstring names before believing it.**

**Proving a guard test "bites" means identifying WHICH assertion fails under the mutation, not that
the test fails.** A multi-assert test can fail for a reason unrelated to the defect it was written
for: one k-NN tie-break test "verified the gate" by deleting a carry flag and seeing 10 of 12 cases
fail — but only the *index* assert failed, distances stayed bit-identical, and the docstring already
declared the identity of a tied neighbour unspecified. So the test pinned a convention the public API
disclaims and cost a second kernel path per bucket. Corollaries: if a test compares triwarp against
itself, ask what an external oracle would say instead; and **before building a test around a plan's
failure-mode claim, reproduce the claim**.

**A Class C threshold with a large headroom is a threshold that does not bite, and the probe is
what shows it.** §7.4 asks for a margin of at least 3x *between the threshold and the measured
agreement*, which is a floor against flakiness; it says nothing about the ceiling, and four of the
tests probed in one pass sat an order of magnitude or more above their own agreement. Two were
retightened on the probe's own numbers, which is what makes a 10 % offset error or a 5 % scale error
fail where only a much larger one did before. The probe to run is the one that re-runs the
*reference* on a deliberately wrong input, not one that perturbs the triwarp side: it measures what
the threshold can actually distinguish. Two results that went the other way and are recorded as
such: `heat_geodesic`'s 5 % bar against the exact great-circle field cannot be tightened, because a
third of it is genuine discretization error in both methods; and the two `heat_signed_distance`
correlations pair with an error bound that has only **1.19x** headroom on `hemisphere`, so that one
is at its floor already.

**A rank correlation is scale-invariant and an error bound is not, so a Class C test carrying both
is carrying two different guards — say which catches what.** Measured on `heat_signed_distance`:
shuffling one side fails the error bound but leaves the correlation at 0.73 on a 12-vertex fixture;
negating one side fails the correlation and leaves the error bound's magnitude untouched; scaling
one side by 1.5 fails the error bound and leaves the correlation *exactly* unchanged. Neither
statistic alone excludes the bug class.

**Run the mutation probe on a test whose whole point is a boundary predicate.** A branch-agreement
test for `greedy_downsample_mask` did not bite: mutating the search's `>=` to `>` left all three
spacings (uniform / clustered / duplicate points) passing, because none put two points *exactly* one
step apart. "Random inputs plus a ties case" does not reach an exact tie — a fourth spacing of 0.25
segments with step 1.0, both exact in float32, does.

### 7.5 Shared helpers — check both modules before writing a private one

Reuse `tests/comparisons.py` (`lexsort_rows`, `assert_unordered_rows_equal`, `undirected_edges`,
`edge_multiplicity`, `euler_characteristic`, `open_edge_count`, `canonical_labels`, `same_partition`,
`canonical_winding`, `assert_same_up_to_sign`, `assert_cyclic_permutation_equal`,
`assert_same_loop_set`, `trimesh_outline_loops`, `fraction_within`, `symmetric_chamfer`,
`chamfer_two_sided`, `symmetric_surface_distance`, `hausdorff_two_sided`,
`hausdorff_surface_two_sided`) and `tests/conversions.py` (`numpy_to_warp`, `numpy_to_warp_uv`,
`points_to_warp`, `points_to_warp_uv`, `trimesh_to_warp`, `warp_to_trimesh`, `trimesh_to_open3d`,
`points_to_open3d`, `open3d_to_trimesh`, `trimesh_to_open3d_t`, `trimesh_to_pymeshlab`,
`warp_to_pymeshlab`, `points_to_pymeshlab`, `trimesh_to_pyvista`, `points_to_pyvista`,
`pyvista_edges_to_indices`, `numpy_to_meshlib`, `trimesh_to_meshlib`, `warp_to_meshlib`,
`points_to_meshlib`, `meshlib_to_trimesh`, `numpy_to_meshlib_bitset`, `meshlib_scalars_to_numpy`,
`meshlib_indices_to_numpy`, `meshlib_bitset_to_numpy`, `numpy_to_pymeshfix`, `trimesh_to_pymeshfix`,
`warp_to_pymeshfix`, `pymeshfix_to_numpy`, `pymeshfix_intersecting_faces`, `pymeshfix_face_remap`,
`points_to_torch`, `numpy_to_pytorch3d`, `trimesh_to_pytorch3d`, `warp_to_pytorch3d`,
`points_to_pytorch3d`, `pytorch3d_to_numpy`, `faces_igl`, `mesh_igl`) rather than re-rolling either.
Every one of the six helpers consolidated in one pass was written by someone who did not check, and
`undirected_edges` alone had been spelled three different ways across six files.

**`points_to_warp` and `warp_to_trimesh` are the two most-reached-for, and both were re-rolled for a
long time.** The bare-cloud upload had been written out **403** times in four equivalent spellings
across 35 files *plus* six one-line private copies, because every reference library had a
`points_to_*` and Warp did not. The readback direction is the mirror image and the asymmetry is
worth knowing about yourself: **a test author reaches for the shared helper when *building* the
reference and writes the readback by hand, every time**.

`canonical_labels` is the label-packing transform every component comparison needs — triwarp names a
component after a representative element, igl and scipy number `0..k-1` in their own traversal orders
and VTK's `RegionId` numbers them in a third, so only the *partition* is shared.

**Two of the class-C helpers take different inputs and the mesh one raises on point arrays.**
`symmetric_chamfer(mesh_a, mesh_b)` takes two *meshes* and samples them itself, where
`chamfer_two_sided(points_a, points_b)` takes two clouds already drawn — which is what a comparison
between two *samplers* needs. **Prefer the mean form over `hausdorff_two_sided` where the claim is
distributional**: on two independent samplings of `icosphere(2)`, the mean statistic separates the
same mesh from one scaled by 1.15 by **6.8x** where the worst-case Hausdorff separates them by
**1.4**, because one stray sample in a tail dominates a maximum. `symmetric_chamfer` also has a
sampling noise floor — a mesh against itself does not score zero — so a threshold must clear that.

**`lexsort` is unusable on float coordinates with ties.** `lexsort_rows` sorts exactly, so two sides
that tie in `float32` but differ in the 16th digit in `float64` order those rows differently and the
compare fails by the full coordinate range. Both are false negatives. For **positions**, match with a
`cKDTree` nearest-neighbour query plus a bijection check, or use `hausdorff_two_sided`; keep
`lexsort_rows` for integer index rows, where it is exact.

### 7.6 The nine reference libraries

**All nine are hard test dependencies. Import them plainly — never through `pytest.importorskip`.**
That rule covers every package in `[dependency-groups] test`, not only the reference libraries:
`libigl`, `shapely` and `moderngl` were guarded by eight `importorskip` calls, which is a latent hole
rather than a safety net — if the import breaks the tests *vanish* instead of failing, and one of
those eight sat under a `parity` marker that `tests/test_parity.py` would have kept passing because
the marker is static and the skip is not. Where a *runtime* precondition genuinely cannot be declared
as a dependency (an EGL driver for `moderngl`'s OpenGL reference), keep a skip, but make it name the
driver and put it in the fixture that needs the context, never at module scope over the import.

Aliases are pinned in ruff's import conventions: `import trimesh as tm`, `import igl`,
`import potpourri3d as pp3d`, `import pymeshlab as ml`, `import open3d as o3d`, `import pyvista as
pv`, `from meshlib import mrmeshpy as mm` / `mrmeshnumpy as mn`, `import pymeshfix` /
`from pymeshfix import _meshfix`, `import pytorch3d.ops as p3d_ops` / `.loss` / `.structures`.

**Threading, because it decides how a ratio reads:** single-threaded — trimesh, igl, pyvista,
pymeshfix (verified: wall clock equals `process_time`). Multi-threaded — `meshlib` (over a hundred
OS threads measured live) and `pytorch3d-cpu`. GPU — `pytorch3d-cuda` only. A `triwarp-cpu` row
loses to a threaded reference on any parallel op regardless of algorithm, which is §9's "decide on
the CUDA number" again.

There are **nine** libraries here and **ten** subsections: the last one, `promesh`, is not one of
the nine and is not a reference at all — it is a design mirror that nothing installs, and it is
filed here only because that is where a reader looks for it.

#### libigl (`igl`)

The reference whose input convention matches triwarp's most closely — `float64` `(n, 3)` vertices and
`int64` `(n_faces, 3)` faces, which is exactly what `mesh_tm.vertices` / `mesh_tm.faces` already are
— and every bound function is *pure*. The exceptions are the stateful solver objects
(`HeatGeodesicsData`, `ARAPData`, `min_quad_with_fixed_data`, `AABB`), which cache a factorization
and must be constructed **inside** a timed callable. Five hazards, all measured:

- **An out-of-range face index is a SIGSEGV, not an exception.** `igl.cotmatrix(V, F)` with one entry
  of `F` past `len(V)` kills the interpreter with exit code 139 and no traceback — igl
  bounds-checks nothing. Never hand it a reduced `V` with the original `F`. The same class of crash
  hits `igl.principal_curvature` on a non-manifold vertex, and `igl.heat_geodesics_precompute` /
  `igl.harmonic` / `igl.lscm` refuse (raise) rather than crash on meshes they cannot factor.
- **Three bound functions are memory-unsafe on ordinary input, so a "works" probe is not enough** —
  check *values*, and prefer a fixture class where the function is known safe. `igl.loop` aborts with
  `free(): invalid pointer` on a five-vertex mesh with three faces on one edge and SIGSEGVs on
  `bunny_decimated`; on `bunny` it silently returns one `NaN` row per unreferenced vertex, because it
  indexes `igl::adjacency_list` (sized `F.max() + 1`) up to `n_verts`. **`igl.in_element` is unusable
  outright**: on a two-triangle square it never reports element 0 for any query inside it, the same
  query returns a face in one batch size and `-1` in another, and a 200-point Delaunay input aborts
  with `malloc(): invalid size` — use `scipy.spatial.Delaunay.find_simplex`, or pyvista's
  `find_containing_cell`. And **`igl.upsample` corrupts the process heap on the scan meshes**, which
  matters more than the other two because the SIGSEGV lands *later*, in unrelated code, and
  `--benchmark-json` is written at session end — so it silently destroyed every row of
  `benchmarks/test_remesh.py` for two measurement rounds. One selection per process: the igl rows
  crash some of the time with the igl rows alone and *every* time with other libraries co-resident,
  and per mesh almost always on the small meshes and never on the largest. Two readings to *not*
  take: it is not an interaction with triwarp, and it is not a size limit. Compacting the
  unreferenced vertices away makes it worse. It is safe on `icosahedron`, so it stays a tested
  reference there and is *not* a benchmarked one.
- **F-only functions size their output by `F.max() + 1`, not by `len(V)`.**
  `igl.adjacency_matrix`, `igl.vertex_components` and `igl.is_vertex_manifold` return `F.max() + 1`
  rows where `igl.cotmatrix` and `igl.gaussian_curvature` return `len(V)`. So on a mesh with
  unreferenced vertices the two families disagree with each other and only the `(V, F)` family
  matches triwarp; a comparison against the F-only family is class B with the transform named. Also
  `igl.connected_components(igl.adjacency_matrix(F))` counts every isolated vertex as its own
  component.
- **Several call signatures are not what the docs suggest, and two fail silently.**
  `igl.exact_geodesic(V, F, vs, vt)` returns an *empty array* rather than raising, because
  `VS/FS/VT/FT` all default to `array([])` and a 4-argument call binds `vt` to `FS` — pass all six,
  with the face arrays explicitly `np.array([], dtype=np.int64)`.
  `igl.knn(P, V, k, *igl.octree(V)[:4])` takes seven positional arguments. `igl.in_element` needs a
  live `igl.AABB`. `igl.crouzeix_raviart_*` need `(V, F, E, EMAP)` from `igl.unique_edge_map(F)`.
  `igl.average_onto_vertices`'s `S` is a per-face *scalar*; `igl.cut_mesh`'s `C` is a per-corner
  **bool** mask, not an edge list.
- **`collapse_small_triangles` and `resolve_duplicated_faces` are not bound**, despite the C++
  headers existing and `triwarp.repair` carrying functions named after them (`AttributeError`).
  Generally: the C++ surface is ~493 headers and only 150 functions are bound, so confirm a name
  exists in the wheel before planning a comparison around it. A hand port of the C++ into a test file
  is a legitimate *test* oracle (see `tests/test_metrics.py`, `test_polyline.py`, `test_seams.py`)
  but never a benchmark row.

**Licensing:** libigl's core is MPL2, but everything under `reference/libigl/include/igl/copyleft/`
is **GPL** — the CGAL boolean suite, `progressive_hulls`, `quadprog`, tetgen and
`copyleft/marching_cubes`. No triwarp code may be derived from that subtree; read the MPL2 top-level
`marching_cubes.h` if a reference is needed, never the `copyleft/` one.

#### potpourri3d (`pp3d`)

For the heat-method family, tangent spaces and isocontours — where neither trimesh nor igl has an
equivalent. pybind11 over geometry-central; takes `float64` `(n, 3)` vertices and `(n_faces, 3)`
`int32` faces.

- **Construct solvers with the non-default flags** `use_robust=False` /
  `use_intrinsic_delaunay=False` so both sides discretize the same triangulation; the defaults
  mollify and flip to an intrinsic Delaunay triangulation first.
- **Tangent-space quantities are gauge-dependent.** Frames agree only up to a rotation about the
  normal, and transport angles only through gauge-invariant combinations (the holonomy around a
  face). Never compare 2D tangent components or single connection phases element-wise.
- **Barycentric output must be decoded.** `marching_triangles` returns `(element_index, coords)`
  pairs in geometry-central's *own* element numbering — decode edges through `pp3d.edges(V, F)`,
  dispatching on `len(coords)` (0 = vertex, 1 = edge, 2 = face). Its closed curves repeat their first
  point; open ones do not.
- **It rejects some inputs outright**, with a `RuntimeError` rather than a wrong answer:
  `MeshVectorHeatSolver` / `GeodesicTracer` / fast marching need a manifold mesh, and `pp3d.edges`
  needs every vertex referenced by a face. Pick fixtures accordingly instead of catching the error.
- A zero cotangent weight (an edge whose two opposite angles are both right angles, i.e. every quad
  grid split by a diagonal — `cave_cube`, `half_torus`) erases that edge's phase from
  `get_connection_laplacian()`, so it cannot serve as an oracle there at all; see
  `tests/test_tangent.py`.

#### pymeshlab (`ml`)

The broadest reference — 281 filters. Build the MeshSet with
`tests.conversions.trimesh_to_pymeshlab(mesh_tm)` (or `warp_to_pymeshlab`) rather than hand-rolling
`ml.MeshSet()`. Use it where it is a *better* oracle than the incumbent, not everywhere. Every one of
these traps **fails green** — a row that reports the build, the `k=1` solve, or a zero-hole no-op
looks like a plausible measurement:

- **Almost every filter mutates `current_mesh()` in place.** `apply_coord_*` moves vertices,
  `meshing_*` rewrites the topology, `compute_*_per_vertex` writes an attribute, `generate_*` pushes
  a *new* mesh onto the set. So one MeshSet serves one filter call; build a fresh one per comparison.
  `compute_curvature_principal_directions_per_vertex` and
  `meshing_decimation_quadric_edge_collapse` additionally default to `autoclean=True` and will delete
  unreferenced vertices under you. The build is a real per-vertex cost, so the MeshSet goes inside
  the timed callable — with one measured exception, the selection filters, whose cost is independent
  of how much is selected.
- **`get_*` filters return a dict; `compute_*` / `meshing_*` / `apply_*` return `None`** (or a small
  dict of statistics) and the answer must be read off `current_mesh()` — `vertex_matrix`,
  `face_matrix`, `vertex_normal_matrix`, `vertex_scalar_array`, `face_scalar_array`,
  `vertex_selection_array`, `face_selection_array`, `vertex_curvature_principal_dir{1,2}_matrix`,
  `edge_matrix`. Selections come back as bool arrays and scalars as float arrays.
- **Length parameters take a wrapper type.** `ml.PercentageValue(1)` is 1 % of the bbox diagonal;
  `ml.PureValue(x)` is an absolute length (this version has no `AbsoluteValue`). Feed `PureValue`
  from the same number triwarp gets.
- **`harm_function` is a no-op.** `compute_texcoord_parametrization_harmonic` returns bit-identical
  texture coordinates at `harm_function=1`, `2`, `3` (max deviation exactly 0.0) and identical
  timings, where libigl's own `k=2` costs several times its `k=1`. Never map triwarp's `k` onto it.
- **Four defaults silently measure nothing.** `meshing_close_holes(maxholesize=30)` closes *zero*
  512-edge rims; `get_hausdorff_distance(samplenum=8)` samples 8 points of the whole cloud;
  `generate_sampling_poisson_disk(radius=0%)` autoguesses instead of using yours; and
  `generate_surface_reconstruction_ball_pivoting(clustering=0)` reconstructs **nothing** and returns
  *faster* for it. Always assert on the returned dict and assert the reference produced output
  before comparing to it.
- **`get_hausdorff_distance` has a *second* silent default: `maxdist` returns `inf`.** It defaults to
  a percentage of the bbox diagonal, and a pair separated further comes back with `min` / `max` /
  `mean` / `RMS` of `inf` — no exception, no warning — until `maxdist=ml.PureValue(1e6)` is passed.
  Its `min` is a sound *upper bound* on the surface-to-surface minimum and is tight whenever a
  sample lands on the witness — always, for a convex pair, because the support point of a
  **polytope** is a vertex. It is a registered oracle for `mesh_to_mesh_distance` on that basis.
- **Some parameters are silent no-ops, and one runs backwards.**
  `compute_scalar_by_shape_diameter_function_per_vertex`'s `cone_amplitude` produces byte-identical
  output at 90 and 120 degrees; `apply_normal_smoothing_per_face` and
  `apply_scalar_smoothing_per_vertex` expose no parameters at all; and
  `generate_resampled_uniform_mesh`'s `offset` as a `PercentageValue` runs from *full erosion* at 0 %
  to full dilation at 100 %, so its own 50 % default is the **zero** offset — pass `PureValue(0.0)`
  when you mean zero. **Probe a parameter before building an axis on it.**
- **Non-manifold input raises, it does not degrade.** `meshing_surface_subdivision_midpoint` and
  `generate_polyline_from_planar_section` both fail on *every* scan mesh. Same boundary as libigl /
  potpourri3d. Other preconditions that raise:
  `compute_texcoord_parametrization_{harmonic,lscm}` need a *boundary* (a closed mesh fails — use
  `hemisphere` / `half_torus`); `compute_matrix_by_fitting_to_plane` needs `set_selection_all()`
  first; `compute_matrix_by_icp_between_meshes` needs **both** layers to carry faces;
  `face_face_adjacency_matrix()` raises `MissingComponentException` unless the FF component was
  requested (`update_topology()` alone does not enable it). `generate_boolean_*` takes `first_mesh` /
  `second_mesh`, not `first` / `second`.
- **Selection morphology is face-based.** `apply_selection_dilatation` / `..._erosion` dilate the
  *face* set; a vertex selection handed to them is simply cleared. From a vertex seed the bridge is
  `compute_selection_transfer_vertex_to_face(inclusive=False)` — `inclusive=True` (the default)
  selects only faces whose *every* vertex is selected. Dilate then maps exactly onto
  `expand_vertex_mask`; **erode does not** map onto `shrink_vertex_mask`.
- **`generate_surface_reconstruction_vcg` returns 0 faces** on an oriented cloud at every voxel size
  probed. Rejected as a reference.
- **Two filters differ from triwarp's port by definition, not tolerance**, and the tests say so:
  `apply_scalar_smoothing_per_vertex` averages a *boundary* vertex over its two boundary neighbours
  alone (so its oracle runs on closed fixtures), and `apply_coord_two_steps_smoothing` at its own
  defaults moves a noisy cube *further* from clean than the noise was, because its fitting step
  rounds corners in.
- `MeshSet(verbose=False)` is the default but several filters print to stdout regardless (ICP, the
  point-cloud normal estimator, the VCG reconstructor); pytest's fd capture absorbs it.
- **`face_normal_matrix()` after `compute_normal_per_face()` is the *unnormalised* cross product**
  (magnitude exactly `2 * area`), so it checks normals *and* areas.
- **Pass counts are conventions**: MeshLab's Taubin `stepsmoothnum` counts lambda-mu *pairs* where
  triwarp and trimesh do one half-step per `iterations`, so the mapping is `2 * stepsmoothnum`; and
  `get_scalar_statistics_per_vertex`'s `"med"` is the sorted element at index `n // 2 - 1`, one
  *below* the middle, for both parities.
- **MeshLab has two uniform coordinate umbrellas, and neither is documented.** Recovered by solving
  least-squares for the per-vertex stencil over 12 random position sets on one connectivity (residual
  2e-16): `apply_coord_laplacian_smoothing` and `apply_coord_unsharp_mask` weight each neighbour by
  its shared-face count and include the vertex itself once (`1/(2d+1)` self, `2/(2d+1)` per neighbour
  on a closed mesh), while `apply_coord_taubin_smoothing` uses the plain 1-ring mean. The difference
  is 8 % of the displacement — far too large to read as a tolerance. **That technique is the general
  one: one pass of a linear filter is a linear map, so its stencil is solvable.**
- **MeshLab writes a layer transform, not vertices.** `compute_matrix_by_icp_between_meshes` leaves
  `vertex_matrix()` byte-identical to the input; the answer is `transform_matrix()` /
  `transformed_vertex_matrix()`. Read the wrong one and a converged ICP looks like a no-op.

**Licensing:** pymeshlab is GPL. `triwarp/` may name it in prose (it does, throughout); no triwarp
code may be derived from its source.

#### open3d (`o3d`)

Build meshes with `tests.conversions.trimesh_to_open3d` (reusable across calls, unlike a MeshSet),
clouds with `points_to_open3d`, and tensor-API meshes with `trimesh_to_open3d_t` — never chain off an
unbound `from_legacy(...)`. The installed wheel is a CUDA build whose legacy `open3d.geometry` /
`open3d.pipelines` APIs are CPU-only; only `open3d.t` has GPU kernels. Nine hazards, all measured:

- **Legacy `remove_*` / `orient_*` / `filter_*` methods mutate in place** (build inside the timed
  callable); the pure `compute_*` / `get_*` / `is_*` calls recompute unconditionally and can share
  one mesh. The trap in the second family: **`get_volume` validates before it integrates**, and the
  validation is the full brute-force `IsWatertight` composition — seconds, on a mesh whose integral
  is microseconds — and it *raises* on non-watertight input. Never benchmark it as "volume".
- **Open3D's tensor meshes must be held in a name.**
  `o3d.t.geometry.TriangleMesh.from_legacy(x).fill_holes()` lets the temporary be collected and the
  result reads freed memory — garbage floats rather than an exception.
- **`fill_holes` winds its cap against the rest of the mesh**, so a raw signed volume of its output is
  meaningless. `trimesh.repair.fix_winding` first.
- **k-NN distances come back squared** from both `KDTreeFlann` and `o3d.core.nns`. Use
  `o3d.core.nns.NearestNeighborSearch` for anything batched (indices match `scipy.spatial.KDTree`
  byte-for-byte on a tie-free cloud); the legacy tree's only query is a per-point Python loop, many
  times slower at scale. `KDTreeFlann`'s radius search is **exclusive at exactly `r`** where
  triwarp's ball queries are inclusive — random clouds never tie, so only a constructed fixture can
  expose it.
- **`is_vertex_manifold` tests connectivity, not a fan**: three faces sharing one edge pass it and
  fail triwarp's and igl's fan definition. The answers agree exactly on edge-manifold input —
  restrict the comparison to that class and pin the divergence. `is_edge_manifold` shares triwarp's
  `allow_boundary_edges` switch with identical semantics.
- **Smoothing filters re-derive inverse-distance weights from current positions every pass**
  (`filter_smooth_laplacian`, `filter_smooth_taubin`), so they match triwarp's fixed assembled
  operator at one iteration and diverge over ten; Taubin's `number_of_iterations` counts lambda-mu
  *pairs*. `filter_sharpen` adds `strength * (deg(v) * v - Σ neighbours)` — the *unnormalized*
  residual, so its displacement is triwarp's times the vertex degree and no parameter mapping fixes
  an irregular mesh. All three are D2 exemptions.
- **Platonic solids come in rotated frames and odd scales**: the octahedron matches triwarp's vertex
  table exactly, but the tetrahedron is rotated and the icosahedron is the raw `(0, ±1, ±φ)` table —
  compare rigid-motion invariants after scaling to unit circumradius, never positions. There is no
  `create_dodecahedron`.
- **`RaycastingScene.compute_signed_distance` shares triwarp's convention exactly** (negative inside,
  parity-ray sign) — no negation, unlike trimesh. But `compute_closest_points` diverges at
  equidistant-face ties, so compare *distances*, not the returned points.
- **`get_oriented_bounding_box` is PCA of the hull and minimizes nothing** (well above triwarp's
  volume on a tilted half_torus); the comparable entry point is `get_minimal_oriented_bounding_box`.
- **`remove_radius_outlier` is nondeterministic**, so it cannot be a class-A oracle. It shares one
  `KDTreeFlann` across an `#pragma omp parallel for` whose radius search is not thread-safe under
  that sharing: three distinct keep sets over eight repetitions of one 500-point cloud. Its published
  rule (`count > nb_points`, self counted) is sound, and evaluating it through the *same* tree one
  query at a time reproduces `points.radius_outlier_mask` exactly. So the comparison goes through
  `search_radius_vector_3d` in a loop and the filter keeps only the benchmark row. No other
  `remove_*` method shares the defect, which is why this one had to be found rather than assumed.
- **A down-sampler's output order is its own, not its algorithm's.** Every legacy selection routes
  through `SelectByIndex`, which walks a *mask* over the input and emits survivors in ascending index
  order — so `farthest_point_down_sample`'s greedy sequence is destroyed on the way out and only the
  selected *set* can be compared. Where the order is the claim, transcribe the C++ loop into the
  test: `FarthestPointDownSample` takes its arg-max with a strict `>`, so the lowest index wins a
  tie. Also `num_samples=0` returns an empty cloud rather than raising, and
  `compute_nearest_neighbor_distance` reports **`0.0`** for a cloud of fewer than two points where
  the honest answer is `inf`.
- Benchmark gotchas: `remove_duplicated_triangles` mutates in place, returns `self` and is
  idempotent; `compute_vertex_normals` overwrites but never caches (a shared mesh is fine);
  `select_by_index` takes **vertex** not triangle indices; `Vector3dVector` rejects non-writeable
  arrays, so use `np.array(...)` not `np.ascontiguousarray(...)` on a trimesh `TrackedArray`;
  `fill_holes` is tensor-API only.

#### pyvista (`pv`)

The **VTK** reference (VTK 9.6 through pyvista's own layer). Build meshes with
`tests.conversions.trimesh_to_pyvista` and clouds with `points_to_pyvista`; one `PolyData` serves many
comparisons because pyvista caches nothing. `pyvista.core` needs no renderer and every comparison runs
headless. A `uv run` whose working directory is inside `reference/pyvista` resolves *that* project and
builds a second virtualenv — always invoke probes from the repo root. Licence: pyvista is MIT and VTK
is BSD-3, so there is no `copyleft/` subtree to avoid. Fifteen hazards, all measured:

- **Float64 in, sometimes float32 out.** Point storage is exact and `regular_faces` is a real
  `(n, 3)` `int64` array — that is why pyvista, not vedo, is the VTK oracle. But `compute_normals`'
  `Normals`, `ray_trace`'s hit points, `fit_plane_to_points(return_meta=True)`'s centre and normal
  and `texture_map_to_*`'s coordinates come back **float32**, while `multi_ray_trace`,
  `principal_axes`, `curvature` and `compute_implicit_distance` are float64. Check the dtype per
  row. Where a difference of large numbers is taken (the angle defect), use `atol` and know the
  residual is **triwarp's** float32 vertex buffer.
- **Nothing is cached, and one filter of twelve mutates.** Repeat calls recompute, so a shared
  `PolyData` is right. The exception: **`edge_mask` writes `point_ind` into its input.** Every filter
  with an `inplace` switch defaults to `False`; never pass `True`.
- **An extraction renumbers its points.** `extract_feature_edges` returns a new `PolyData` carrying
  only the points its lines touch, in its own order — on a unit box it keeps all 8 and still orders
  them differently, so a comparison that skips the remap fails on a mesh whose *counts* match and
  reads as a real disagreement. Go through `tests.conversions.pyvista_edges_to_indices`.
  `extract_cells` / `extract_points` / `threshold` / `clip_box` return an `UnstructuredGrid` (index
  maps in `vtkOriginalPointIds` / `vtkOriginalCellIds`); `split_bodies` / `bounding_box` /
  `oriented_bounding_box` return a `MultiBlock` unless passed `as_composite=False`; and
  `remove_points` / `collision` / `ray_trace` / `contour_banded` return **tuples**.
- **`cell_quality` is 12 usable measures of 28 on triangles, with two naming inversions and one
  constant.** pyvista's `radius_ratio` is triwarp's `aspect_ratio`; triwarp's `radius_ratio` is its
  reciprocal; `shape` is triwarp's `mean_ratio` and `aspect_frobenius` is one over it; `condition`
  duplicates `aspect_frobenius` exactly; `min_angle` / `max_angle` are in **degrees**; `distortion`
  is a constant **1.0** on ordinary input, so a threshold test on it passes on anything; and the 16
  measures that do not apply come back as the `-1.0` null value rather than raising.
  `tests/test_triangles.py` carries the decoding.
- **`is_manifold` is `n_open_edges == 0`, and `n_open_edges` counts boundary *plus* non-manifold
  edges.** So `is_manifold` maps to `is_edge_manifold(allow_boundary_edges=False)` and the *count*
  does not map to `boundary_edges`: on three faces sharing one edge it reads **7** where triwarp
  counts 6 boundary edges. Likewise **`DataSet.center` is the bounding-box centre**, not a centroid,
  and `bounding_sphere` returns `(radius, center)` — radius first — and is a genuine near-minimal
  sphere, not the sphere about the AABB centre; the two coincide on any centrally symmetric mesh
  (which is how the wrong reading survived three fixtures) and differ materially on a hemisphere.
- **`curvature('maximum')` / `('minimum')` are algebra, not estimation** — exactly
  `H ± √(H² − K)` from VTK's own Gauss and mean curvature, max abs difference **0.0** in float64, and
  complex on the many icosphere vertices where `H² < K`. They are not an independent
  principal-curvature implementation. `curvature('gaussian')` is exactly the angle defect over the
  barycentric lumped area and is a genuine oracle.
- **Surface operators return surface quantities.** `compute_derivative`'s gradient is the
  *tangential* one — mean `2/3 · e_x` for `f = x` on the unit sphere, which triwarp's
  `face_gradients` reproduces after `average_onto_vertices` — and it stays on the **points** for a
  point-data field even with `preference='cell'`.
- **Both smoothers are different algorithms, not different tunings.** `smooth` moves each vertex
  along its incident edge directions under VTK's own convergence test, so it diverges with the
  iteration count against a fixed assembled operator; `smooth_taubin` is the windowed-sinc filter and
  warns *"An optimal offset for the smoothing filter could not be found"* on ordinary input. Its
  iteration count is in lambda-mu **pairs**. Both are D2 exemptions.
- **Several answers are empty, constant or unchanged rather than wrong, so assert non-vacuity
  first.** `clip_surface(pv.Sphere(radius=0.6))` returns **0 cells** on `icosphere(3)`;
  `extract_values(0.010, scalars='area')` returns 0 cells (it matches values *exactly* — a range is
  `ranges=`); `edge_mask(30)` is all-`False` on a smooth sphere (use a box); `integrate_data` of a
  symmetric point field reads ~0; `validate_mesh().coincident_points` is **empty** on ten
  exactly duplicated vertices (`clean` is the dedup oracle, and its `zero_size` — not
  `degenerate_faces` — is where a zero-area triangle lands); `lines_from_points` gives one two-point
  line cell per segment rather than one polyline, so `compute_arc_length` restarts every segment and
  `decimate_polyline` is a no-op at every reduction; `tube` / `ribbon` emit triangle **strips**
  (`n_faces == 0` — `.triangulate()` first); and `extrude(capping=True)` leaves 16 open edges.
- **Where the reference put the answer, and four deprecated names.** `align(return_matrix=True)`
  returns `(aligned_mesh, 4×4 matrix)` and *does* move the points (unlike MeshLab); `geodesic` puts
  the ordered path in `vtkOriginalPointIds` and its Euclidean length equals `geodesic_distance`;
  `sample` marks misses with `vtkValidPointMask` *and* a `vtkGhostType` array;
  `voxelize_binary_mask` writes a **point** array named `mask` on a cell-centred grid, so it is
  *solid* and its set is contained in triwarp's `mode="solid"` answer rather than equal to it.
  Deprecated in 0.48.4: module-level `pv.voxelize` / `pv.voxelize_volume` (a hard
  `DeprecationError`), `select_enclosed_points` (→ `select_interior_points`, array name now lowercase
  `selected_points`), `extract_geometry` (→ `extract_surface(algorithm=None)`) and `n_faces_strict`
  (→ `n_faces`).
- **`multi_ray_trace` is trimesh + embree, not VTK.** It imports `trimesh`, checks
  `trimesh.ray.has_embree` and calls `tmesh.ray.intersects_location` — identical to trimesh's own
  call, so a `pyvista` row for the `intersects_*` groups would be a **trimesh row under another
  name**. VTK's own `ray_trace` *is* independent but takes one ray per call, two orders of magnitude
  off embree, so it is a test oracle and never a benchmark row.
- **`validate_mesh()`'s cell fields are per *cell*, not per mesh.** `intersecting_faces` is "two
  faces of a **3D cell**", so on a triangle mesh it is identically empty — 0 on two interpenetrating
  icospheres where `face_self_intersecting_mask` flags dozens, and 0 on `bohemian_dome`.
  `inverted_faces` likewise reads 0 on a mesh with ten reversed faces. The degeneracy field that does
  fire is **`zero_size`**, and `clean()` **keeps** those faces at every tolerance — so there is a
  detector here and no filter. A degeneracy comparison also needs a *scale-aware* input: a
  float64-exactly-collinear face survives triwarp's float32 altitude test on both devices (§12.4).
  Relatedly, `collision` is a **two-mesh** filter and cannot see a self-intersection either: it
  reports thousands of hits for a mesh against its own copy.
- **`compute_implicit_distance` needs polygons.** On a line-set `PolyData` VTK logs *"No polygons to
  evaluate function!"* once per query and returns a field far off the truth rather than raising.
  Polyline distance goes through `find_closest_cell` on a **single-cell** polyline instead — and that
  single cell is the whole trick: `pv.lines_from_points` makes one cell per segment, which is what
  makes `compute_arc_length` read orders of magnitude short. Its locator collapses on a long cell:
  usable at a few thousand queries against a few hundred segments, minutes at 65 536 of each.
- **`find_containing_cell` is the point-location oracle that works**, batched, `-1` outside, and
  exactly agreeing with `scipy.spatial.Delaunay.find_simplex` on 10 000 queries with the batch and
  the per-point loop byte-identical. Worth stating because igl records `in_element` as unusable for
  the identical question, so generalizing from it skips a good reference. `find_closest_cell` is
  likewise the most accurate closest-point reference registered (float64-exact against
  `igl.point_mesh_squared_distance` on distance *and* point) — but its **cell id is not comparable**:
  it disagrees with igl on 28 % of exterior queries, every one a point lying on a shared edge to
  ~1e-16. Compare the distance, at Warp's own `mesh_query_point_no_sign` floor (§12.4).
- **Two filters answer a *different* question than their name suggests.** `sample()` interpolates
  only where the query lands **inside** a source cell — a fraction of the target points against
  `interpolation.transfer_onto_vertices`, agreeing closely on those — and
  `snap_to_closest_point=True` snaps to the nearest source **vertex**, not the nearest point on the
  surface, which makes it *worse*. And `delaunay_2d(edge_source=loop)` does not clip to the loop: on
  a 40-point star it returns cells covering more area than the polygon. The polygon-fill oracle is
  `triangulate_contours`, which adds zero Steiner points and matches
  `polyline.triangulate_polyline`'s `n - 2` count and area to nine digits.
- **Parametric surfaces arrive open and `clean` defaults differently per surface.**
  `surface_from_para(clean=False)` is the underlying default and at that setting every one of the 21
  has 156 or 236 boundary edges (a raw `ParametricMobius()` is a disk), while pyvista overrides it to
  `clean=True` on 9 of them, so `klein` arrives welded and `mobius` does not. **Always pass
  `clean=True` explicitly.** And **`klein` is not a Klein bottle** as VTK parameterizes it: it welds
  to two boundary loops and reads *orientable*, so only `figure8_klein` is the closed non-orientable
  χ = 0 surface. The mirror is byte-identical to the installed wheel for
  `core/utilities/parametric_objects.py`, so those algorithms may be read from the submodule; but the
  surfaces' **domains and periodicity are not in pyvista at all** — they live in VTK's
  `vtkParametric*` constructors as `JoinU` / `JoinV` / `TwistU` / `TwistV`.

#### meshlib (`mm` / `mn`)

Build meshes with `tests.conversions.trimesh_to_meshlib` / `numpy_to_meshlib` / `warp_to_meshlib`,
clouds with `points_to_meshlib`, and read a result back with `meshlib_to_trimesh` — never hand-roll
`mn.meshFromFacesVerts`, whose argument order is the trap below. It earns its seat on three counts
the other eight cannot cover: it is the **only multi-threaded** CPU reference, so a `triwarp-cuda` vs
`meshlib` ratio is a fair fight; it is the only reference that binds a real minimum-weight
Liepa/Klincsek `fillHole` with a 12-metric family and a two-loop `stitchHoles`; and it is the only
oracle for `repair.collapse_small_triangles`. `meshlib.mrcudapy` exists and is deliberately **not**
used: the plain `mrmeshpy` free functions are the reference, and a CUDA module would make the row
incomparable with the others. Fourteen hazards, all measured:

- **Almost every free function mutates its `Mesh` in place and returns something else.** `relax`,
  `fillHole`, `fillHoles`, `decimateMesh`, `remesh`, `subdivideMesh`, `fixMeshDegeneracies`,
  `filterCreaseEdges`, `denoiseNormals`, `smoothRegionBoundary`, `expand` and `shrink` return a
  status `bool`, a count, an `EdgeId` or a `FaceBitSet` of *new* faces — never the mesh. So **one
  `mm.Mesh` serves one mutating call**: build a fresh one per comparison and per benchmark round
  (`BenchCase.new_mesh_ml()` is a method, not a cached property, for exactly this). The exceptions
  are worth knowing per function rather than assumed: `marchingCubes` reads its `SimpleVolume` and
  returns a fresh `Mesh`, so *its* input is cacheable.
- **`meshFromFacesVerts` takes faces *first*, and it sizes the vertex buffer by `F.max() + 1`.** The
  argument order is the reverse of every other converter in `tests/conversions.py`, and a swapped
  call raises nothing. The sizing behaves *differently* by position: a **trailing** unreferenced
  vertex is dropped outright while an **interior** one is kept in the buffer and excluded from
  `numValidVerts`. So on a mesh whose spares are interior the buffers line up and the *validity* mask
  does not; where the spares are trailing, the indices shift. Never hand MeshLib a compacted `V` with
  the original `F`, and never assume `getNumpyVerts(...).shape[0] == len(V)`.
- **`pack()` is mandatory before reading topology back, and skipping it is silent.** After a
  `decimateMesh` that leaves 120 valid faces of 320, `topology.faceSize()` still reads 320 and
  **`getNumpyFaces` returns `last_valid_face_id + 1` rows**, most of them `[0, 0, 0]`. No exception,
  no warning. `meshlib_to_trimesh` packs by default.
- **`np.asarray` on a scalar container silently returns a 0-d `object` array.** `VertScalars`,
  `FaceScalars` and `UndirectedEdgeScalars` are not accepted by `mn.toNumpyArray` (which binds only
  `VertCoords` / `FaceNormals` / `std_vector_Vector3_float` and raises a clear `TypeError` otherwise
  — that part is safe), and `np.asarray(vert_scalars)` produces `dtype=object, shape=()` rather than
  raising, so the failure surfaces several lines later. Go through
  `conversions.meshlib_scalars_to_numpy` — and for a container of **ids** rather than numbers through
  `conversions.meshlib_indices_to_numpy`, which exists because the scalar reader *raises* on them: a
  `VertId` implements `__index__` but not `__float__`.
- **A returned bitset is only as long as its highest set bit, and which functions do that is not
  guessable.** `mn.getNumpyBitSet` reads the bitset at *its own* length: one MeshLib sized against
  the mesh comes back domain-sized, but one built by insertion does not —
  `findSelfCollidingTrianglesBS` returns a short array on a colliding pair and an **empty** one on a
  clean mesh. Neither raises, and both break an `np.array_equal` by *shape* rather than by value,
  which reads as a converter bug. Go through `conversions.meshlib_bitset_to_numpy(bitset, size)`.
- **Every bitset converts in bulk, in both directions — a per-bit Python loop is never the answer.**
  A `TypedBitSet` derives from `MR::BitSet` and `mn.getNumpyBitSet` is declared over the base, so
  pybind11 upcasts any of them. The *load* direction is `BitSet.fromBlocks`, which takes packed
  `uint64` blocks — so `np.packbits(flags, bitorder="little")` fills a whole set in one call
  (`conversions.numpy_to_meshlib_bitset`; wrap the result in the typed set). Against the per-cell
  `set()` loop it replaces that is two to three orders of magnitude, in both directions. Three
  details: `bitorder="little"` is not NumPy's default and is not optional; `fromBlocks` rejects a
  NumPy `uint64` array with `TypeError` (pass `.tolist()`) and rounds the size up to whole 64-bit
  blocks, so `resize` back to the domain; and the element order is the container's, which for a
  `VoxelBitSet` addressed by a `VolumeIndexer` is `x` fastest, i.e. `dense.ravel(order="F")`. **A
  "MeshLib binds no converter for this" claim is a reason to probe the base class, not to write the
  loop.**
- **`(*args, **kwargs)` in a signature is an overload set, and `inspect` / `help()` cannot see it.**
  This wheel's pybind11 docstrings are stripped. **Read the real signatures by calling the function
  with one junk argument and reading the `TypeError`**, which pybind11 renders as a numbered list of
  every overload.
- **Two overloads of one name can have opposite output conventions, and the wrong one binds
  silently.** `expand(topology, region: FaceBitSet, hops)` returns `None` and **mutates `region`**,
  while `expand(topology, f: FaceId, hops)` **returns** a new `FaceBitSet`; same for `shrink`.
  `stitchHoles(mesh, a, b, params)` takes two named hole edges and `stitchHoles(mesh, params)` finds
  them itself — argument *count* is the only tell. `relax`'s first overload takes a `PointCloud` and
  the second a `Mesh`, with different params types. One `getAllComponents` form returns a
  `(components, count)` **tuple**. Resolve the overload explicitly and assert the result's type or
  count before comparing.
- **`findOutliers`' default mask segfaults on a cloud with no normals.** `FindOutliersParams.mask`
  defaults to `OutlierTypeMask.All`, which includes `AwayNormal`, and that criterion dereferences the
  cloud's normals. The other three modes run fine without normals.
- **A projector stores a raw pointer to the mesh or cloud it was given, so a temporary segfaults.**
  `PointsToMeshProjector.updateMeshData(build_a_mesh())` and
  `PointsProjector.setPointCloud(build_a_cloud())` both return normally and then read freed memory in
  `findProjections`. Bind the mesh or cloud to a name that outlives every query — Open3D's
  `from_legacy` hazard in a second library. **`mm.MeshPart` does *not* share the rule — it keeps a
  real Python reference**, so `mm.MeshPart(trimesh_to_meshlib(mesh_tm))` over a temporary is safe:
  `sys.getrefcount(mesh_ml)` rises across the constructor and `part.mesh is mesh_ml`, where the two
  setters above leave it unchanged. That is the whole test, and it is the one to run on the next
  binding of this shape: **probe the refcount, do not infer the lifetime.**
  `findProjections`'s `upDistLimitSq` is a second crash of the same shape: pass MeshLib's own
  `FLT_MAX`, since `math.inf` segfaults rather than raising.
- **The AABB tree is lazily built and cached on the `Mesh`, and the ratio depends on whether the
  query is the process's first.** First-vs-second `findProjection` is one to two orders of magnitude
  apart, the largest gap being the process's very first mesh while the thread pool spins up. Every
  query row and every timed callable must state whether the build is inside it — recommended: build
  outside and pre-warm with one throwaway query. This is also a *correctness* trap next to the
  mutation hazard: a mutating call invalidates the tree. **A `PointCloud` caches its point tree the
  same way**, and `cloud.invalidateCaches()` in the benchmark's `setup` is the lever.
- **Per-vertex free functions are per-*vertex*, and `mrmeshnumpy` has the batched form.**
  `discreteGaussianCurvature(topology, points, v)` and `sumAngles(...)` take one `VertId` per call; a
  Python loop over them is one to two orders of magnitude slower than
  `mn.getNumpyGaussianCurvature(mesh)`, bit-identical, and the gap grows with the mesh. Never put a
  per-vertex Python loop in a benchmark row.
- **`getNumpyVerts` is float64 but the storage is float32**, so a MeshLib comparison bottoms out
  around 1e-7. `computePerFaceNormals` is **normalized** — note the contrast with pymeshlab's
  unnormalised `face_normal_matrix()`.
- **Four parameter conventions that read as a disagreement, and one function whose name lies.**
  `sampleHalfSphere()` is **not** a half sphere: its directions span `z` from -1 to +1, so feeding it
  to `computeSkyViewFactor` as a sky dome halves the answer. `InSphereSearchSettings.maxRadius`
  defaults to **1** whatever the mesh's scale, silently capping every thickness on anything larger —
  pass half the smallest bounding-box side. `makeUVSphere`'s `verticalResolution` counts interior
  latitude **rings**, not profile points, so it pairs with `creation.uv_sphere(count=(v + 2, h // 2))`
  — and at *that* mapping the two are the same mesh vertex for vertex, where the same nominal
  resolution differs by 74 % in the vertex count. `leftCotan(e)` is the **plain** cotangent keyed by
  the directed edge whose left face owns it, against `laplacian.cotmatrix_entries`' *half* cotangent
  keyed by `(face, corner)`; `cotan(ue)` is the two summed. Two more found the same way:
  `MarchingCubesParams.origin` addresses the voxel **centre**, so a lattice whose sample `[0, 0, 0]`
  sits at `lower` is marched with `origin = lower - voxel / 2` and the un-shifted call is a rigid
  half-diagonal off; and **`findNClosestPointsPerPoint` returns a heap, not a sorted list** — the ids
  are exactly scipy's `k` nearest but only ~91 % of rows are in distance order and the *nearest* is
  the **last** entry, which at `numNei=2` holds in 100 % of rows only because a two-element heap is
  ordered by construction. Ask for `numNei=1` when one neighbour is the question.
- **`computeRayThicknessAtVertices` takes the direction from the *pseudonormal*.** So it pairs with
  `visibility.thickness(method="ray", normals=angle_weighted_vertex_normals(...))` and sits five
  orders worse against the area-weighted normals — the kind of gap that reads as an algorithm bug.
  Both thickness functions and `computeInSphereThicknessAtVertices` also take **no query set**, which
  keeps them out of a benchmark group whose input is a subsample.
- **Its detectors are reliable oracles; several of its mutators are not.** `mm.eliminateTunnels`
  leaves the mesh **byte-identical** on every configuration probed (a 2 048-face torus and a genus-2
  union, every `maxTunnelLength`, `maxIters`, `TunnelLoopType` and the `FillHoleNicelySettings`
  overload) while on that same torus `detectTunnelFaces` and `detectBasisTunnels` both answer
  correctly. And `inflate(mesh, verts, InflateSettings)` takes the **unselected** vertices as its
  Dirichlet condition, so selecting every vertex leaves the system with no anchor and collapses
  `icosphere(3)` to the origin at every pressure; given a *region* it solves a different problem (the
  implicit Laplacian flattens the cap onto its pinned rim, so volume falls before pressure raises
  it). Meanwhile `findSpikeVertices` and `findInnerVertsOfDegree` are exact Class A oracles.
  **Before building a comparison on a MeshLib mutator, run it once and assert it changed the mesh**;
  where it did not, fall back to the invariant that carries the claim (χ arithmetic for tunnels,
  volume monotonicity and normal alignment for inflation) and say in the test docstring what was
  probed.
- **`mm.localFixSelfIntersections` is the third inert mutator, and it needs a *single-component*
  input — the part no signature states.** On two welded copies of one sphere it returns its input,
  `np.array_equal` on both buffers with every colliding triangle still colliding, at every
  configuration probed (both `Method` values, every `relaxIterations`, `maxExpand`,
  `subdivideEdgeLen`, `touchIsIntersection=False`, `mimicPatch=True`). On a single-component
  self-intersecting torus it does mutate — and still does not clear, doubling the colliding count.
  Its sibling `mm.fixSelfIntersections` (the voxel path) has no such limit.

  Two lessons the rule above already states and neither of which was applied here. A benchmark
  callable that returns `numValidFaces()` and asserts `> 0` **passes a no-op**, and this read as the
  suite's largest single loss row for four rounds. And **apply the same detector to both outputs** —
  an earlier "both libraries reduce, and neither clears" reading had counted only triwarp's side.

  **The fix was the fixture, not a skip — the better move whenever a reference declines an input
  rather than being wrong about it.** A skip drops a comparison; giving the group a
  single-component input restored one, and the row is like-for-like for the first time.

  **Its behaviour is fixture-dependent in both directions, so neither reading generalizes**: on a
  trimesh-built torus it clears, on a MeshLib-built one it makes things worse, on two welded spheres
  it declines. Probe a mutator on the *specific* input a row or a test will use.
**Two pairings worth knowing before writing a comparison, neither guessable from the names:**
`computePerVertNormals` matches `vertices.area_weighted_vertex_normals` while
`computePerVertPseudoNormals` matches `angle_weighted_vertex_normals`, each to float32 rounding, and
each sits orders of magnitude further from the other's partner — so MeshLib pins a weighting
convention no other reference distinguishes. And `mn.getNumpyGaussianCurvature` is the pointwise
**angle defect**, which pairs with `vertices.vertex_defects` and **not** with
`curvature.discrete_gaussian_curvature`, the Cohen-Steiner/Morvan *ball* measure.

**Licensing: MeshLib is the one reference here that is not open source.** The wheel and
`reference/MeshLib` are under AMV Consulting's *"NON-COMMERCIAL & education"* agreement — a
terminable, non-transferable licence, with a separate commercial licence required otherwise, and an
explicit bar on modifying or transferring the Software. That is a stronger constraint than libigl's
`copyleft/` subtree or pymeshlab's GPL, because it restricts *use* rather than distribution, and
triwarp ships `MIT OR Apache-2.0`.

**Nothing under `triwarp/` may name MeshLib at all.** Not the library, not a function (`fillHole`,
`positionVertsSmoothly`, `triangleAspectRatio`), not a source file (`MRMeshDelone.cpp`,
`MRTriMath.h`). 89 such references were removed in one pass across 19 files; they had accumulated as
ordinary attribution and collectively read as a claim that a package shipped under
`MIT OR Apache-2.0` is derived from a proprietary one. Describe what the code **computes**, or name
the algorithm in the literature's vocabulary — "the Liepa/Klincsek interval DP", "the Delone
empty-circumcircle test", "circum-radius over twice the in-radius" — which is what a reader needed
anyway. Read `reference/MeshLib` to understand an operation's *interface* and its parameters; never
to port its body. **Keep it a test/benchmark dependency**, where naming it is correct and required.
**Check 21 enforces this** rather than trusting a remembered grep — the rule has failed twice, once
at 89 sites and once at five, two of which said "port of" in the imperative. Its pattern is wider
than four symbols, because those matched **none** of the three file-name comments the eighth kernels
pass found: it keys on `meshlib` / `mrmeshpy` / `mrmeshnumpy`, on an `MR<CamelCase>` prefix
generically, and on the C++ identifiers that carry no `MR` at all (`FanOptimizer`,
`buildLocalTriangulation`, `positionVertsSmoothly`, `calcQueueElement_`, `updateBorderQueueElement_`).
Add to that pattern when a new symbol is found; do not narrow it.

#### pymeshfix (`_pmf`)

nanobind over Marco Attene's MeshFix / the TMesh kernel. Build every `PyTMesh` through
`tests.conversions.numpy_to_pymeshfix` / `trimesh_to_pymeshfix` / `warp_to_pymeshfix` and read one
back with `pymeshfix_to_numpy`, `pymeshfix_intersecting_faces` or `pymeshfix_face_remap` — the
converters exist because the raw calls are unsafe in three separate ways.

It is the **narrowest deep** reference: `dir(PyTMesh)` is 19 members, of which **nine are
algorithms** (`fill_small_boundaries`, `select_intersecting_triangles`, `strong_degeneracy_removal`,
`strong_intersection_removal`, `clean`, `remove_smallest_components`, `join_closest_components`,
`fix_connectivity`, plus module-level `clean_from_arrays`) against MeshLib's 246 — and all nine are
*repair*, which is where triwarp has 20 public functions. It performs one operation no other
reference does end to end: arrays of a broken digitised surface in, a single watertight solid out.
Do **not** plan a comparison for anything else: there is no curvature, geodesic, parametrization,
registration, reconstruction, decimation, remeshing, point-cloud, boolean, proximity or
signed-distance entry point, and the C++ names for several (`cutAndStitch`, `iterativeEdgeSwaps`,
`loopSubdivision`, `isInnerPoint`, `openToDisk`, `marchIntersections.cpp`) are in the headers and
**unbound**. Thirteen hazards, all measured:

- **`load_array` is not a load; it is already a repair, and it renumbers.** It runs the kernel's
  connectivity fix and Euler update before returning. On `icosphere(1)`: a trailing *or* interior
  unreferenced vertex is dropped; an exactly duplicated face is **kept** and the non-manifold edges
  it creates are cut instead (so both counts *rise*), while a *reversed* duplicate is refused and the
  vertices are still cut; two coincident *referenced* vertices are **not** merged; one backwards face
  is rewound and *every* face backwards is left alone (the signed volume is unchanged), because
  consistent is not outward. The scan meshes gain or lose vertices and faces on load, and on a
  non-orientable closed surface it cuts the orientation-reversing seam, leaving two coincident sheets
  where the surface had one — which is why `select_intersecting_triangles` over-reports there and the
  number is not a disagreement. So **every comparison must be index-free** (positions, canonically
  sorted rows, sets, counts); where a face index is unavoidable, go through `pymeshfix_face_remap`,
  which checks the load first and refuses rather than guessing. The flip side is that `load_array` is
  itself an oracle — for `remove_unreferenced_vertices`, for `make_winding_consistent`, and partly
  for `split_non_manifold_vertices`.
- **One `PyTMesh` serves one load and one mutating call.** A second `load_array` raises
  `RuntimeError`, and every algorithm mutates in place and returns a status, a count or an array.
- **`select_intersecting_triangles` returns a mostly-uninitialised array.** It allocates `(n, 3)`
  `int32` and writes the `n` face indices into the **flat** prefix, leaving `2n` entries of heap
  garbage whose value varies between processes. `out.ravel()[: out.shape[0]]` is the only defined
  read. The prefix *is* deterministic, so a naive `np.array_equal(out1, out2)` reports
  nondeterminism that is not there.
- **`tris_per_cell` and `justproper` are no-ops on ordinary input.** Every combination probed returns
  the identical answer. Pass both explicitly so a wheel that starts honouring either one fails a test
  rather than drifting, and do **not** build a triwarp flag around `justproper`.
- **`nbe` is inclusive, both its docstrings say otherwise, and pymeshlab's counterpart is
  exclusive.** `fill_small_boundaries(nbe, …)` fills loops of **at most** `nbe` boundary edges where
  the C++ comment and the Python docstring both say "less than"; `nbe = 0` means all. This is the one
  place two references disagree about the *same* parameter — pymeshlab fills the same rim only at
  `maxholesize = nbe + 1` — so `holes.fill_small(max_edges=...)` follows pymeshfix (the precedence
  rule below) and a pymeshlab comparison passes `max_edges + 1` as its named class-B transform.
- **The "MeshFix could not fix everything" line on stderr is printed when it *succeeded*.** The
  wrapper does `if (result) cerr << …` where `result` is *true only if the mesh was completely
  cleaned*. The message is inverted, `set_quiet` does not suppress it, and **nothing about it may be
  used as a signal** — read the boolean, or read the mesh.
- **`remove_smallest_components` ranks by face count, not area or diameter, and returns the number
  removed.** On three disjoint spheres — two of 80 faces, one of them at ten times the radius, and
  one of 320 — it keeps the **320-face** one. It always reduces to exactly one component; that is the
  rule `repair.remove_small_components(keep_largest=True)` defaults to.
- **The output face buffer is a reordering, even when nothing was repaired.** `icosphere(2)` round
  trips with **byte-identical float64 vertices** and an identical triangle *set* under
  `np.sort(rows, axis=1)` plus a lexsort, but the rows come back in a different order and each starts
  at a different corner. Never compare face buffers positionally.
- **`n_boundaries` is a property in 0.18.1, and `boundaries()` raises.** The older Cython wheel
  exposed `boundaries()`; the nanobind one keeps the name bound only to raise, and `n_points` /
  `n_faces` became properties in the same change. Code written against an example older than 0.17
  fails with `TypeError: 'int' object is not callable`.
- **`strong_degeneracy_removal` measures degeneracy in `double`, so it is *stricter* than triwarp's
  `float32` test rather than merely different.** On a flat strip: exactly collinear vertices are
  removed by both, and the same strip offset by `1e-9` is removed by triwarp and **kept unchanged**
  by pymeshfix. Where the degeneracy is exact the two agree completely, so compare on an *exactly*
  degenerate fixture and pin the near-degenerate class as the divergence.
- **`strong_intersection_removal` is a different algorithm from
  `repair.fix_self_intersections(method="local")`, not a different tuning**, and no transform rescues
  the pair: on a self-intersecting torus triwarp cuts and refills each sheet and ends with **two**
  closed components where pymeshfix removes far more and ends with **one**. All they share is the
  post-condition, so that pair is neither benchmarked nor a parity claim. The comparable level is the
  whole pipeline: `repair.make_solid` against `clean_from_arrays` agrees to float32 rounding and
  returns the identical vertex and face counts on `bunny_decimated`.
- **Reproducing `clean_from_arrays` needs the loader's repair as an explicit first stage.** It is
  invisible in the C++ pipeline because `load_array` does it, and its absence is invisible in the
  output too until you check the right predicate: without `remove_unreferenced_vertices` +
  `make_winding_consistent` + `split_non_manifold_vertices` first, the result comes back with χ = 2
  and one component and is **not watertight**, because nothing downstream looks at edge manifoldness.
  Two more orderings measured rather than reasoned: the component filter has to run *inside* the
  intersection loop as well as before it (cutting a band out can disconnect the surface), and
  **nothing geometric may run after the final fill** (filling a 3-vertex rim makes one sliver, a
  degeneracy pass deletes it and reopens the rim, and the two trade the same faces for ever).
- **`trimesh.slice_plane`'s output is a poor input** — see §7.3.

**Benchmark rule: on most rows the load *is* the row.** Because a `PyTMesh` takes exactly one load,
the build has to sit inside the timed callable (`BenchCase.new_tmesh_pmf()`), and on the scan meshes
it is most of the round. **Create a `pymeshfix` row only where the operation is at least ~30 % of
the round, and state the measured share in the group docstring.** By that rule the intersection
family and `clean_from_arrays` are timed, and the hole-fill and component-removal comparisons carry
`pytest.mark.parity(<group>, "pymeshfix", benchmarked=False, reason=…)` with the ratio in the reason.
Cap it at `bunny`.

**Licensing: pymeshfix is GPL-3.0**, and the TMesh headers under `reference/pymeshfix/src/` carry
Attene's dual licence — GPLv3 *or* a commercial agreement with IMATI-GE/CNR. This is a **different**
constraint from MeshLib's, and conflating the two over- or under-restricts:

| | MeshLib | pymeshfix |
|---|---|---|
| May `triwarp/` name it? | **No** | **Yes**, in `Notes` / `See Also` — pymeshlab is GPL and is named throughout |
| May `triwarp/` be derived from its source? | No | **No** |
| May `tests/` and `benchmarks/` import it? | Yes | Yes |
| Why | proprietary, restricts *use* | copyleft, restricts *distribution of derivatives* |

Read `reference/pymeshfix/src/` for the interface and the parameters, never for the body; cite the
**paper** rather than the file when porting — Attene, *"A lightweight approach to repairing digitized
polygon meshes"* (The Visual Computer 26, 2010) for the repair pipeline and the component-joining
rule, Liepa, *"Filling holes in meshes"* (SGP 2003) §3 for the density refinement, Barequet & Sharir
(1995) for the loop pairing — and keep `grep -rnE 'MeshFix|Basic_TMesh|TMesh|_meshfix' triwarp/`
empty. Prose mentions of `pymeshfix` itself are allowed there; C++ symbols are not.

**And the precedence rule, which is the part a future author most needs and would never guess:**

> **Where pymeshfix and MeshLib both answer a question and their answers differ, triwarp's default is
> pymeshfix's answer and MeshLib's is reachable by a flag** — not the reverse. Where only MeshLib
> answers it, nothing changes.

The reason is not preference. A function whose *behaviour* was pinned against MeshLib alone is a
function whose specification lives in a proprietary binary nobody may read; pymeshfix's source is
mirrored and readable by anyone. The one place this currently bites is `holes.fill_small`, which
takes a **perimeter** threshold because MeshLib's `fillHoles` does, where pymeshfix and pymeshlab
both take a **boundary-edge count**

#### pytorch3d (`p3d_ops` / `p3d_loss` / `p3d_structures`)

Build meshes with `tests.conversions.trimesh_to_pytorch3d` / `numpy_to_pytorch3d` /
`warp_to_pytorch3d`, clouds with `points_to_pytorch3d` (the `loss` container form) or `points_to_torch`
(the bare batched tensor the `ops` entry points take), and read back with `pytorch3d_to_numpy`.

It is the **first and only reference with CUDA kernels of its own**, so it takes *two* `LIBRARIES`
rows (`pytorch3d-cpu` / `pytorch3d-cuda`) and a `triwarp-cuda` vs `pytorch3d-cuda` ratio is the one
GPU-against-GPU comparison in the suite. It is also the first reference triwarp **already cited in
its own prose without testing against**: `metrics.py`, `registration.py` and `kernels/metrics.py`
named it nine times, so two public functions had their specification pinned to a library nothing in
the suite ran. Fifteen hazards, all measured:

- **Everything is batched, with a leading minibatch axis, and the wrap fails silently.**
  `knn_points(p, q)` handed a bare `(P, 3)` reads it as `(N=P, P1=3, D)` and compares three points at
  full speed — no exception, and a *faster* wrong answer. A single cloud goes in as `x[None]` and its
  answer comes out as `result[0]`, which is what `points_to_torch` is for, and every comparison
  asserts the reference's output **shape** before its values.
- **Every neighbour and Chamfer distance it returns is squared.** `knn_points().dists`,
  `ball_query().dists`, `loss.chamfer_distance` and both `point_mesh_*` scalars. Take the square root
  before comparing (measured **0.0** afterwards for `knn_points` on the host, 1.19e-07 on CUDA where
  its own kernel is a different reduction order).
- **`torch.cuda.is_available()` is the wrong probe for the CUDA extension.** A wheel whose arch list
  stops short of the device returns `True` and then fails every kernel with *"no kernel image is
  available"*; a build that compiled the CPU extension only — which is what `setup.py` selects
  whenever it finds no `CUDA_HOME`, the normal state of a CI runner — raises `RuntimeError: Not
  compiled with GPU support.` on a CUDA tensor. `benchmarks/conftest.py` gates its `-cuda` row on
  launching a two-point `knn_points`, which is the only probe that distinguishes them.
- **`Meshes` and `Pointclouds` are immutable *caching* containers** — the opposite end of the scale
  from a `ml.MeshSet`. Every `ops.*` / `loss.*` entry point is pure, so **one object serves many
  comparisons**; but `verts_normals_packed`, `edges_packed`, `faces_packed_to_edges_packed`,
  `laplacian_packed` and `faces_areas_packed` are memoized on first request, so a *benchmark* row
  naming one has to build the container **inside** the timed callable or it reports nothing. And the
  answer lives on the `*_packed()` accessors, never in the constructor arguments.
- **It does not cast for you, and one entry point casts anyway.** A float64 `Meshes` keeps float64
  through `verts_packed()`, so an unconverted reference compares a float64 answer against triwarp's
  float32 one and reads as triwarp being wrong by ~1e-7 — hence float32 in the converters. The
  exception is `ops.mesh_face_areas_normals`, whose C++ kernel returns **float32** whatever it was
  handed. Faces are stored as int64 regardless.
- **`corresponding_points_alignment` and `iterative_closest_point` are row-vector.** They solve
  `s·X·R + T = Y`, so `R` is the transpose of `registration.procrustes`' linear block divided by the
  scale (2.54e-07 / 2.62e-07); `T` needs no transform. Compare the **converged transform and the
  rmse**, never the iteration count.
- **`ops.cot_laplacian` returns two conventions in one call and neither is guessable.** Its
  off-diagonal is **twice** triwarp's half-cotangent table and its diagonal is identically **0.0**
  (not merely small) where `laplacian.cotmatrix` assembles the row sum; and its second return is
  `1 / inv_areas == 3 * M_ii`, the *reciprocal* of three times the barycentric lumped mass. Both
  cancel out of `mesh_laplacian_smoothing`'s ratios. `ops.laplacian` writes **-1** on the diagonal
  where `laplacian.laplacian(equal_weight=True)` writes 0, and `ops.norm_laplacian` **is**
  `laplacian(equal_weight=False)` before its row normalization — same formula, same literal
  `eps = 1e-12`, measured 1.49e-08 after dividing by the row sum. **It also never coalesces** — see
  §16.6 for why the `cotmatrix` ratio is a scope mismatch and the rewrite it invited was declined.
- **`ops.marching_cubes` does not exist**: it is not re-exported from `pytorch3d.ops`, only from
  `pytorch3d.ops.marching_cubes`. With `return_local_coords=False` it emits lattice indices, which is
  `levelset.marching_cubes`' own default and needs no convention fix at all — the one reference of
  five that does not.
- **`ops.cubify`'s three `align` modes are one uniform scale and translation apart.** On a 6³
  occupancy sphere, `"topleft"` / `"corner"` / `"center"` return the **identical** face buffer and
  vertex count with bounding boxes `[-0.6, 1.0]`, `[-0.667, 0.667]` and `[-0.8, 0.8]` — all three
  reachable through `voxels.from_cells(cells, voxel_size, origin)`, so an `align=` keyword would be a
  second spelling of one that exists. It also **compacts**, which is what exposed
  `to_boxes(cull_internal=True)` returning the whole corner lattice.
- **`packed_to_padded` / `padded_to_packed` bounds-check nothing and corrupt the heap.** A mismatched
  `max_size` or `total_size` does not raise — it writes out of bounds and the process dies in
  `malloc`/`free` much later. Their `first_idxs` are **starting indices, not counts**, which makes
  them `array.pack_1d_arrays`' `offsets` unchanged (measured `[0, 3, 8]` from both sides for lengths
  `(3, 5, 2)`); read the sizes off the buffers rather than passing literals.
- **`sample_points_from_meshes` caps the face count at 2²⁴.** It draws the face index with
  `torch.multinomial`, whose category limit is 16 777 216, so `lucy`'s 28 055 742 faces raise
  `RuntimeError` rather than sampling. `happy_buddha`'s 1 087 716 are fine.
- **`add_points_features_to_volume_densities_features` is `[-1, 1]` local space, `[z, y, x]` storage,
  and `rescale_features=True`.** Its volume is `(minibatch, channels, D, H, W)` and a point's
  `(x, y, z)` indexes `(W, H, D)`, so its lattice is the **transpose** of `voxels.splat_onto_grid`'s;
  `align_corners=True` is triwarp's `bounds`; and the default `rescale_features` divides by
  `density.clamp(min_weight)`, which makes both sides an *average* rather than an accumulation. With
  those three lined up the two agree at exactly **0.0** on the host and 4.8e-07 on CUDA.
- **`mesh_normal_consistency` counts pairs, not adjacencies.** It enumerates every pair of faces
  sharing an edge, so an edge with `k` incident faces contributes `C(k, 2)` terms;
  `adjacency.face_adjacency` keeps only edges with *exactly* two faces. The two agree exactly on
  edge-manifold input (0.0155947 over 480 pairs) and read **0.777 against 0.0** on three faces
  sharing one edge.
- **`ops.taubin_smoothing` rebuilds its operator every half-pass**, from the current positions, and
  row-normalizes it — where every triwarp smoother assembles once. One fixed operator sits
  4.1e-03 / 6.5e-03 / 9.9e-03 from it at 1 / 3 / 10 of its iterations, the size of the displacement
  itself; `smoothing.filter_taubin(recompute=True)` closes that to 2.4e-07 at **~23x** the cost. Its
  `num_iter` counts lambda-mu **pairs**.
- **`ico_sphere` is *not* the rotated-frame hazard open3d's Platonic solids are.** It starts from the
  identical `(±0.5257, ±0.8507, 0)` table `creation.icosphere` uses and subdivides the same way, so
  the positions correspond one-to-one and the 5.8e-05 residual is pytorch3d's table being *written*
  to four decimal places. Match by nearest vertex with a bijection check at 1e-4, not by index.
  `utils.torus`, by contrast, takes the **minor** radius first and builds its vertex table in a
  Python double loop, so it needs a real parameter mapping and its benchmark column is a per-vertex
  Python floor.
- **Every `laplacian_matrices` entry point builds a COO tensor, which makes torch warn once per
  process — and the two suites answer it in opposite directions on purpose.** torch validates no
  sparse-tensor invariant by default and says so the first time one is built (*"Memory errors (e.g.
  SEGFAULT) will occur ... explicitly opt in or out"*, raised from `ATen/Context.cpp`). It is a
  statement about torch's global state, not about our input, so §7.7's "read the reference's source
  at the warned line and solve for what input makes it fire" does not apply — no input silences it,
  only an explicit choice. It was the **only** warning a CPU test run raised and the only one a
  benchmark run raised. `tests/conftest.py` calls
  `torch.sparse.check_sparse_tensor_invariants.enable()`, beside the `STRICT` launch mode and for
  the same reason: pytorch3d is pinned to upstream `main`, and a malformed tensor from a future pin
  should raise naming the invariant rather than land as a SIGSEGV with no Python frame (§12.1 on
  what misattributing one costs). `BenchLibrary.run` calls `disable()` on a `pytorch3d` row
  instead, because the checks cost **1.04-1.07x** of a sparse construction (interleaved, one
  process, 40 962-vertex `ops.laplacian`) and `laplacian` / `cot_laplacian` / `norm_laplacian` /
  `mesh_laplacian_smoothing` all build theirs *inside* the timed call — charging the reference for
  validation triwarp's row does not perform would bias `test_laplacian`, the one module whose point
  is a like-for-like race between two sparse assemblies.

**Its CPU rows are Θ(N²) on anything with a neighbour query**, because `_C` carries no spatial
structure on *either* device. Measured: `knn_points` 299.8 / 1 189.5 / 4 562.8 ms and
`chamfer_distance` 557.8 / 2 319.1 / 9 077.1 ms at 10 k / 20 k / 40 k self-queries — 3.84-4.16x per
doubling, extrapolating to ~3.5 s per `knn` round on `bunny` and **~9 minutes per round** on
`dragon`. So a `pytorch3d-cpu` neighbour or chamfer row is capped at a feature mesh; the `-cuda` rows
run the scan meshes.

**And the GPU ratio is a crossover, not a bar — the most useful thing this reference measures.**
Brute force with perfect coalescing *beats* a BVH descent while the whole problem still fits the
device's bandwidth: 2.31 ms against triwarp's 3.29 at 20 000 points (`knn_points`, **0.70x — a loss
for triwarp**) and 74.65 against 0.87 at 200 000 (**85x**). So the neighbour and chamfer groups need
the point count as an **axis**; a one-size row reports whichever side of the crossover it landed on.

**Every CUDA row must synchronize torch's stream.** `wp.synchronize_device` synchronizes *Warp's* and
says nothing about torch's, so a `pytorch3d-cuda` row synchronized the Warp way times the launch and
not the kernel. `BenchCase.run` branches on `kind == "pytorch3d"` for exactly this. **And
`torch.cuda.empty_cache()` belongs in that teardown** — torch never frees device memory, and 16
triwarp rows once failed to allocate 65 368 bytes on a 32 GB card after an uncapped pytorch3d row
(§15.4).

**Licensing: pytorch3d is BSD-3 (Meta) and torch is BSD-3**, so there is no subtree to avoid, no use
restriction and no copyleft. `triwarp/` may name it and may be derived from it with attribution. It
stays a test and benchmark dependency all the same: nothing in the shipped package imports torch, and
it must not start, since `torch` is a 2.5 GB install for a library whose whole premise is Warp.

#### promesh — not a reference library, and not citable from `triwarp/`

`reference/promesh/` is a bare source drop, and it is the one mirror here whose terms **cannot be
read from this repo at all**: no `LICENSE`, no `COPYING`, no `pyproject.toml`. It is also not a
published package — nothing installs it, `import promesh` fails, and it therefore can never be a
tested or benchmarked reference, so none of §7.6's machinery applies to it. Treat it as a design
mirror only, on the same footing as reading MeshLib for an *interface*.

**Nothing under `triwarp/` may name it**, for a reason that is the mirror image of MeshLib's rather
than the same one: MeshLib's licence is readable and restricts *use*, where promesh's is simply
unknown — so a "port of" comment cites terms nobody here has checked. Five such references had
accumulated in `holes.py`, two claiming a port outright (*"the Warp port of promesh's
`triangulate_boundaries`"*, *"Mirrors promesh's private helper"*). All five were rewritten into the
algorithm's own vocabulary — the gap-bridging problem of **Barequet & Sharir (1995)**, solved by the
minimal-perimeter heuristic with a longest-increasing-subsequence monotonicity correction — which is
what a reader wanted in the first place. **Check 21 covers both libraries in one scan** (keying
additionally on `promesh` and `triangulate_boundaries`) because the fix is identical either way,
and its own test pins the *replacement* wording as a negative so a later pass cannot widen the
pattern back onto the literature's vocabulary.

One mitigating fact worth knowing before re-reading that tree: `reference/promesh/deformation.py`
attributes its own mollification helper to `kentechx/HoleFillingPy` (**MIT**), so parts of it are
themselves ports of permissively licensed code — which is why the corroborating citation in
`kernels/laplacian.triangle_inequality_slack` names that upstream and not promesh.

### 7.7 Where the reference put the answer

A reference that looks like it *disagrees* is more often one whose result was read from the wrong
place, or which was handed something other than what you thought. Two general rules, each the
generalization of several measured cases above:

- **Ask what a new reference was handed, and whether it finished the job, before reading a ratio.**
  Two instances in one round: `chamfer_backward` (the *inputs* differed) and `cot_laplacian` (the
  *output* is a different object — an uncoalesced COO tensor with no diagonal, §16.6).
- **A reference's zero is not always "off"**, and a reference's warning is usually about *our* input.
  A `RuntimeWarning: invalid value encountered in divide` from `trimesh/triangles.py:659` was
  reported as "reference-library numerics on a stochastic reconstruction" and was **wrong**: that
  line divides by `d1 - d3`, which expands to the squared length of edge AB, so it is a zero-length
  edge and nothing else — `screened_poisson(point_weight=0.0)` was returning 27-64 zero-area
  triangles, every run. Two things made the wrong reading comfortable: the stack frame was entirely
  inside the reference (nothing in the traceback named triwarp), and the warning was *intermittent*
  (3, then 1, then 0 across three suite runs), which reads like reference flakiness — but it was our
  output that varied, and **the degenerate faces were present every single run**; only whether a
  query point's nearest triangle happened to be one of them varied. **Read the reference's source at
  the warned line, solve for what input makes it fire, then assert that property on the buffer we
  passed** — that check is a better detector than the warning, because it fires on every run, and it
  is the right regression guard (`face_nondegenerate_mask`, not `-W error::RuntimeWarning`).

---

## 8. Tooling and local validation

Two tools are configured in `pyproject.toml` and are the authority for style and typing — do not
hand-roll equivalents or add competing tools. Run them, and the tests, after any change to `triwarp/`
and before considering work done.

### Ruff — lint + format

Ruff is the **only** linter and formatter (no black/isort/flake8). Config lives in `[tool.ruff]`; do
not override it inline.

```bash
uv run ruff format triwarp tests      # format (100-col, skip-magic-trailing-comma)
uv run ruff check --fix triwarp tests # lint + autofix
```

- Enabled rule families and per-file ignores are in `[tool.ruff.lint]` — respect them; do not add
  blanket `# noqa` to silence a rule the config selects. If a rule is genuinely wrong for a
  construct, prefer a scoped `per-file-ignores` entry over inline suppression. The one standing
  exception is §1.6's kernel accumulator, which needs `# noqa: UP018, RUF046`.
- Import conventions are enforced (`[tool.ruff.lint.flake8-import-conventions.aliases]`) — see §7.6
  for the reference-library aliases.
- Docstrings are enforced (`D`), so every public function needs one (§6).
- `reference/` is excluded from both lint and type-checking — never edit vendored code to satisfy a
  tool.
- **Ruff's `F811` is a dead-code check, not a collision check.** After merging modules (or their test
  / benchmark files), two same-named definitions collide silently and only one is linted: a duplicate
  `def` trips `F811` **only if the first binding is unused**; a duplicate `def` whose first binding
  *is* used before the redefinition is **invisible** and still changes behaviour, since module-level
  `def`s all execute at import so the *last* definition wins for every call site including those
  written above it (two `_sources_wp` helpers in `benchmarks/`, one cached and one not, silently
  dropped the cache); and duplicate module-level **constants** are never linted at all. **After any
  module merge, run an AST scan for repeated top-level `FunctionDef` / `ClassDef` / `Assign` names,
  and confirm the test count is unchanged across the merge** — that is the check that catches a
  silently deleted test.

### basedpyright — type checking

basedpyright (a Pyright superset) is the configured type checker — it is what the IDE runs, so match
its verdict. Config lives in `[tool.basedpyright]`.

```bash
uv run basedpyright
```

- **`triwarp/kernels/` is excluded.** The Warp kernel DSL is not modeled by any stubs and is
  inherently un-typecheckable. Do not try to make kernels type-clean or add `# pyright: ignore` there.
- **Warp-stub type-flow rules are disabled** — and the list is exactly three:
  `reportArgumentType`, `reportAttributeAccessIssue` and `reportOperatorIssue`. **`reportReturnType`
  and `reportCallIssue` are both ON** (see below); `reportIndexIssue` and `reportGeneralTypeIssues`
  are *not* disabled either and run at `standard`'s default — read the config, not the prose.
  Warp's Python-scope stubs are weak (`wp.empty` typed as returning a bare `warp.array`, i.e.
  `array[Unknown, int]`; `wp.array.__getitem__` typing every *slice* as `indexedarray | array`;
  `warp.sparse` returning `BsrMatrix[BlockType[...]]`; `BsrMatrix.offsets/.columns/.values` absent
  from the stub), so these rules fire almost entirely on false positives. **Consequence:
  basedpyright will *not* catch genuine argument or index type errors in wrappers** — rely on the §7
  regression tests for correctness, not the type checker. Returns are the exception and are checked.
  It also cannot see a cross-module private call that goes through an attribute path
  (`tw.holes._mean_rim_edge_length`).
    - **`reportReturnType` is ON and is the one of the seven worth paying for.** `wp.empty`'s
      element type is `Unknown`, which is assignable to *anything*, so a `wp.empty` return already
      satisfies a `wp.array[wp.int32]` annotation and **the dtype is never what fails**; believing
      otherwise costs a wasted pass. What fails is **`NDim`, which is invariant**:
      `wp.array[wp.float32]` and `twt.Array1dFloat32` (`array[float32, Literal[1]]`) are not
      assignable to one another in *either* direction, so the package's two rank-1 spellings cannot
      mix in one return. Enabling it reported 56 and cost 42 narrowings, and it earned them by
      catching **five annotations that were simply wrong about what the code returns, none of them
      reachable by any test** — three `laplacian` entry points that declared a `float32` return
      whatever dtype they were handed, a `BsrMatrix[wp.float64]` declared for a system assembled
      from `wp.mat22d` tangent blocks, a `wp.float32` declared for a Python `float`, and one
      function with no return annotation at all, so three callers' declared returns were unchecked.
    - **`reportCallIssue` is the second exception and is ON, at a cost of 44 fixes.** It only ever
      fires where the callee is **overloaded** — the identical defect against a non-overloaded
      callee lands in `reportArgumentType` and stays invisible, which is why
      `polyline.polyline_radius` could pass one buffer to `reduce.mean` (overloaded, resolves via
      its `wp.array[wp.bool]` arm) and to `reduce.median` (not overloaded, silent) on consecutive
      lines. Of the 44, **34 were `triwarp.reduce` calls**, failing only because the overloads
      pinned the rank while their callers use the package's usual rankless `wp.array[dtype]`
      spelling. The fix is a convention worth knowing — **accept wide, return narrow**: a
      *parameter* takes the `Any`-ranked `twt.ArrayNd*` family, a *return* keeps the
      `Literal`-ranked `twt.Array1d*` / `Array2d*` aliases. The dtype still discriminates, so the
      overload set stays resolvable; only the rank check is given up, and only where the callee
      never depended on it. `reduce`'s `axis=` overloads deliberately keep `Array2dScalar`.
    - **Widening an overload's parameters makes `wp.empty` bind *silently* instead of erroring,
      so the two halves are one commit.** `wp.empty`/`wp.zeros` return `array[Unknown, int]`, and
      `Unknown` satisfies every arm — so once the rank stops rejecting it, such an argument picks
      the **first** overload and `reduce.max` on a `float64` buffer infers `int` (13 of the 34 were
      that shape). The fix is [`twt.empty_1d`][triwarp.typing.empty_1d], which carries both dtype
      and rank; reach for it whenever a buffer feeds `reduce`. Where the buffer is also *returned*
      from a `wp.array[dtype]`-annotated function it cannot be converted at all (`NDim` is
      invariant), and takes a `cast` at the call instead. **Check what a buffer is returned as
      before converting its allocation.**
    - **Two smaller findings the pass surfaced.** `reduce`'s public overloads declared only
      int32/float32/float64 while `kernel_reduce._GLOBAL_DTYPES` also carries int64/uint32/uint64 —
      so a `wp.int64` reduction was supported at runtime and undeclared, with two in-repo callers
      relying on it; the global overloads now span it and the `axis` ones deliberately do not. And
      a `wp.vec3`/`wp.mat44` value can never satisfy `wp.normalize` / `wp.transform_point`: their
      stubs annotate the `Vector`/`Matrix` hint shells, which nothing concrete derives from
      (`wp.vec3` is `vec3f`, based on `ctypes.Array`). A value coming *out* of another builtin
      resolves, because `wp.cross` is declared as returning `Vector[...]`; one held in a variable
      never does. **The fix is a `TYPE_CHECKING`-only redeclaration that binds the builtin itself
      at runtime** — `triwarp/typing.py` carries `normalize` / `cross` / `dot` / `transform_point`,
      each `wp.<name>` in the `else` branch, so a call costs exactly what it did and the kernel
      source is untouched. It is not a blanket `Any`: returning a `vec2` from a `-> vec3` function
      still errors. Four narrower spellings were probed and **all four fail** (a `cast` on the
      argument, an annotated local, declaring the *parameter* as the shell, and widening only
      `transform_point`'s point); `n: Any = v` passes and was declined, since it erases every check
      on the name. **Fixing one builtin pushes the gap to the next** — narrowing `normalize` to
      `wp.vec3` immediately broke `wp.dot`, which is why `dot` is in that table too. When upstream
      takes concrete types, delete the block and the `_V` TypeVar with it.
    - **The narrowings are two shapes, and only one of them is a `cast`.** A Python-scope *slice*
      is always a dense `wp.array` — `wp.array.__getitem__` carries no annotations, so pyright
      infers `indexedarray | array` from its body — and `twt.as_dense` narrows it with a real
      `isinstance`, because `wp.indexedarray` is **not** a `wp.array` subclass. Reach for it at a
      slice; a gather (`src[indices]`) genuinely *is* an `indexedarray` (§3.4) and is
      materialized with `wp.copy` instead. Everything else — `warp.sparse`'s
      `BsrMatrix[BlockType[...]]`, `wp.array.list()`, `wp.normalize`/`wp.cross` — is a plain
      `cast`. Counts live in `pyproject.toml`'s comment; re-measure after a `warp-lang` upgrade.
- **`reportPossiblyUnboundVariable` is kept as an error** — it catches §1.4's conditional-scope
  gotcha. When it fires on a *correlated* condition (two separate `if is_mesh:` blocks), fix it the
  way `triwarp/registration.py` does: initialize to `None` before the branch and
  `assert x is not None` at the use site. Do **not** suppress it.
- The gate is expected to stay at **0 errors**. An error is almost always either a real
  possibly-unbound bug or a missing dependency — resolve it, do not widen the disabled-rule
  list.

### Full environment

Test/reference dependencies live under `[dependency-groups] test`, **not** `[project] dependencies`
(triwarp itself only needs `warp-lang`). There is no `[tool.uv] default-groups`, so a bare `uv sync`
does **not** install them and will uninstall trimesh/pytest:

```bash
uv sync --all-groups
```

`uv add <pkg>` puts a package in main `dependencies` by default — use `uv add --group test <pkg>`.
Prerelease pins need `--prerelease allow`; trimesh is pinned `>=5.0.0rc1` because the crack-free
`trimesh.remesh.subdivide_to_size` (the reference for `triwarp.remesh.subdivide_to_size`) only exists
in the 5.0.0rc prereleases — stable 4.x is the older T-junction "soup" variant.

**`include` covers `triwarp`, `tests` and `benchmarks`, so a dev-only env is no longer merely
noisy — it is unusable.** `uv sync --group dev` alone yields hundreds of `reportMissingImports`
across the reference stack, not the handful `meshio` used to produce. Either `uv sync --all-groups`
before running the bare command, or run `uv run basedpyright triwarp`, which is what CI's fast
`typecheck` job does — an explicit path overrides `include`. The wide gate runs in `pytest-cpu`,
the only job whose environment already has all nine reference libraries.

`tests/` and `benchmarks/` are checked under a per-directory `executionEnvironments` block that
concedes the five rules the reference stack and `wp.array.numpy()` make unactionable there
(`reportMissingTypeStubs`, `reportMissingTypeArgument`, `reportCallIssue`,
`reportGeneralTypeIssues`, `reportIndexIssue`), each carrying its measured count in the config.
**An execution environment's `root` re-bases import resolution**, so each entry needs
`extraPaths = ["."]` — without it 240 imports break and the conceded stub errors are merely
replaced by `reportMissingImports`.

Widening `include` cost 938 errors, which the `executionEnvironments` block took to 151 and the
fixes below to 0. `triwarp/` measured 0 at every one of those tiers, so widening cannot regress
the shipped package's gate. Of the 151: **73 were ignore comments the checker itself called
unnecessary** — the tree carried 83 directives and 74 of them sat in `tests/`/`benchmarks/`, which
nothing had ever checked, so they suppressed nothing and most were wrong about the error they
named. The tree now carries **3**, each verified load-bearing by
`reportUnnecessaryTypeIgnoreComment`. **Set the rule set first and then delete what the checker
flags** — the unnecessary count is a function of the rules, so a list taken before the config lands
is the wrong list.

**`plans/` is gitignored** — plan documents are local working notes and never appear in a commit.

### Version control

**Commit directly on `main` unless the user asks for a branch.** This is a single-developer
repository: there is no review queue for a feature branch to sit in and no second working copy to
integrate with, so a branch-and-merge cycle buys nothing and costs a merge. The default overrides
any general "branch before committing" habit — branch only when the user names one, or when a change
is genuinely speculative and expected to be thrown away.

Two things this does **not** change. Committing is still an explicit request: finish the work, run
the gates (§8, §12.10), and commit when asked, not on your own initiative. And the measurement
discipline that depends on a clean tree is unaffected — an A/B against a prior revision uses a
**detached worktree**, never `git stash` and never a branch checkout in the live tree (§15.6).

### Coverage — `pytest --cov`, and `kernels/` is omitted on purpose

`pytest-cov` is in the `test` group and configured in `pyproject.toml`'s `[tool.coverage.run]`.
CI's CPU job measures it, publishes the figure to a Gist that shields.io renders as the README
badge, and *then* enforces the floor — in that order, deliberately, because gating first would
abort a regressing `main` build before the badge updated and freeze it at the last good number.
Three things about it that are decisions rather than defaults:

- **`triwarp/kernels/` is omitted, and this is measured, not tidiness.** A `@wp.kernel` /
  `@wp.func` body is never *called* as Python, so coverage.py reports a kernel that runs on every
  test as entirely unexecuted — the numbers are §12.6. Never "fix" a low kernel-module figure by
  writing tests at it; the same module tree basedpyright and `docs/gen_ref_pages.py` already
  exclude, for the adjacent reason.
- **The badge is the CPU wrapper layer.** No GPU on a runner, so every `device.is_cuda` branch and
  everything behind `_device.prefers_tiled_reduction` is unreachable there. Do **not** raise the
  floor to chase those lines — they are §7.2's two-process `tests.devices` job.
- **There is no `exclude_also`, and the two obvious candidates were tried and refuted.** All 88
  `@overload` stubs write `...` on the `def` line — which *does* execute at import — and there is
  not one bare `...` line under `triwarp/`; and coverage.py 7.16 already excludes an
  `if TYPE_CHECKING:` block by itself (`analysis2` reports 2 excluded lines in `triwarp/io.py`
  under an empty config). Both patterns moved the statement count by exactly **0**. A pattern that
  matches nothing is worse than none, because it reads as a caveat someone has already handled —
  verify an exclusion by diffing the statement count, not by reading the regex.

Coverage is a gate on the *wrapper* layer's branches — validation, `Literal` menus (§4.2), empty
input early returns. It says nothing about whether a comparison is vacuous, which is §7.4's job.

---

## 9. Performance work: measure before you change

The methodology is here; the *numbers* are Part II (§13 cost model, §14 kernel-shape verdicts,
§15 measurement traps, §16 component status).

- **Something suddenly slow is a Warp rebuild until proven otherwise — check that first.** Before
  profiling anything, before believing a kernel got slower, before deleting a test for being slow.
  See §15.1 for the two cheap confirmations and the measurements.
- **A benchmark lands before the optimization does.** Never restructure code for speed without a
  `benchmarks/test_<module>.py` group timing the *current* implementation first. A belief about where
  the cost sits ("two Python loops", "too many derived launches") is a hypothesis until that group
  exists and prints a number.
- **Attribute a cost with *one measurement of the thing itself*.** A projection built by subtracting
  two measurements of *different* things is a hypothesis, and in this repo it has been optimistic by
  **3-10x every time** (§15.2).
- **Attribute at the *benchmark's own* operating point.** A kernel's share of the cost can reverse
  the sign of an optimization between two points on the same axis — grep the benchmark for how it
  derives its parameter and profile at *that* value (§15.3).
- **Read device time before diagnosing anything.** Six of triwarp's ten biggest losses are 92-99 %
  host-side launch/allocation cost, not slow kernels (§16.1).
- **But first ask whether the function graph-captures, because `wp.timing_begin` cannot see a
  replayed kernel and will report a device-bound function as ~100 % host** (§15.10). That covers
  triwarp's own five capture sites *and* every CG solve, since `warp.optim.linear` captures by
  default. Three functions were attributed backwards this way. Disable the capture, measure there,
  and carry the device total back.
- **Before proposing an optimization, read what *calls* the thing — the decline may already be
  written there.** A scan reads bodies; a measured decision is prose, and it lives at the call site,
  in a constant's comment, or in the benchmark's docstring rather than in the function. All three
  items worked in one plan round were refuted by text already in the file the item proposed changing
  (§15.5). Grep the caller, the constant and the benchmark docstring for a number first.
- **Attribute a change only with a back-to-back A/B in one session.** Saved baselines drift ±10 %
  (±30 % under 100 µs) between sessions, so a comparison against a stored number proves nothing.
  Run old and new in the same session, in the same process order — and **use a detached git worktree,
  never `git stash`** (§15.6).
- **Interleave A and B in one loop, and read the `min`.** Timing all of A then all of B lets GPU
  clock state decide the winner. Report the `min` alongside the median: a one-off Warp kernel compile
  or a scheduler hiccup inflates a median but cannot deflate a minimum (§15.7).
- **Verify values, not just timing.** This is §3.4's gather warning generalised: the *wrong*
  implementation is frequently the faster one, because it reads less. Every perf change must keep its
  parity / regression test green, which means a function about to be optimized needs one first.
- **CUDA is the target; a CPU regression is an acceptable price for a GPU win.** triwarp exists to
  run geometry on the GPU, so when the two devices disagree, **decide on the CUDA number.** Do not
  reject a device-side change because Warp's CPU backend is slower at it — Warp's CPU reductions run
  ~1 lane per block while NumPy's are vectorized C, so a host readback plus NumPy wins on CPU almost
  every time and would veto nearly every reduction if it were allowed a vote. Still *measure* both
  (the CPU path must stay correct, and the ratio belongs in the comment), and still decline a change
  that wins nowhere. Worked example: §14.5. **"No gain on CUDA" is the reason to decline, not "slower
  on CPU."** The exception where the *device* becomes part of the branch condition is an algorithm
  that does asymptotically more work to expose parallelism — see §14.7.
- **A share that falls as the input grows is a decline, not a small win.** The saving is largest
  exactly where the call is already cheap. Report the share at more than one size before deciding; a
  single operating point cannot show the trend.
- **A decline is a result — write it at the site, with the number, and resolve every site the finding
  named.** The exemplary case is `kernels/neighbors.py`, where the `wp.length_sq` swap was measured,
  declined, and the reasoning written into the source; the counter-case is the same finding's other
  two sites in `ball_pivoting`, which were neither converted nor annotated and were re-derived a pass
  later. A finding that names five sites is closed when all five are converted **or** annotated.
- **A written decline can expire because a *neighbour* got faster.** `greedy_downsample_mask`'s own
  comment declined a rewrite and said *"revisit if a benchmark ever puts it on top"*; a
  level-synchronous RDP then took `polyline_simplify` from 84 ms to 0.68 ms and **inverted the
  module's ordering**. Re-read the declines in a module after any big win in it.
- **A mechanism validated on one member of a fixture pair built to isolate a variable must be
  measured on the *other* member before it ships.** `saddle`/`saddle_graded`, `sphere`/`tangle`,
  closed/open, uniform/graded — the pair exists because that variable matters, and a benchmark
  suite that ships one is telling you which variable a new mechanism has to survive. This is §7.4's
  vacuity rule ("check the comparison is not vacuous on its fixture") applied to a *benchmark*
  rather than a test, and it has cost a measured 2.59x regression once: a block conjugate gradient
  was probed on two well-conditioned saddles, won on both, shipped, and was a **0.386x** loss on the
  graded saddle of identical connectivity — which is precisely where its own documented failure mode
  lives (§16.8). The check is cheap: it is one more row in the probe you are already running.
- **A decline written in one place is not a decline applied everywhere, and the tree's own source is
  the census you should not trust.** `kernels/reduce.py::_reduce_1d_tiled` has documented "the
  generic form is declined here, measured ~18 us per launch" since it was written — and 34 of that
  module's kernels took the generic default anyway, because the factory's `dtype` parameter had one
  (§16.0). **When a finding is a property Warp computes rather than one the source spells, take the
  census from the runtime**: an AST scan of annotations found 39 of the 71 generic kernels, and
  `wp.get_module(name).kernels` found all 71.
- **A probe that instruments the thing it measures must carry a third, do-nothing arm.** A
  monkeypatched `wp.launch` reported two wrappers *regressing* under a change that was in fact worth
  1.15-1.28x on them: the patch added its own Python frame to every launch in one arm only, and
  silently failed to specialise non-array generic parameters. Adding a "same wrapper, no
  substitution" control separated the two, and a detached-worktree A/B (§15.6) settled it. **Instrument
  both arms identically, or measure without instrumenting.**
- **Re-probe a tuning constant after a Warp upgrade.** The hole-DP block-size reading *"measured flat
  between 32 and 128"* was taken on Warp 1.16 and does not hold on 1.17.
- **Sweep both devices before writing a single tuning number**, and if the two optima differ by more
  than a tolerance, split the constant rather than averaging (§13.3).
- **Count the launches a numerical-method change adds per iteration before proposing it.** A cycle
  here is launch-bound, not flop-bound; any smoother that buys iterations with launches loses
  (§14.8).

---

## 10. Warp API reference mirrors

Authoritative Warp function lists are mirrored locally under `reference/warp_api/`:

| File | Scope |
|------|-------|
| `reference/warp_api/builtins.md` | Built-ins usable inside `@wp.kernel` / `@wp.func` (`wp.<name>`) |
| `reference/warp_api/warp.md` | `warp` module API at Python scope |
| `reference/warp_api/sparse.md` | `warp.sparse` BSR/CSR matrix API |
| `reference/warp_api/utils.md` | `warp.utils` Python-scope utilities |
| `reference/warp_api/fem_linalg.md` | `warp.fem.linalg` linear-algebra utilities |

BEFORE using an unfamiliar Warp builtin, sparse, or utils function, `grep` these files to confirm the
exact name, signature, and scope rather than guessing. Each file stamps the Warp version it was
transcribed from, and its source URL, at the top — fetch the URL for full argument details when the
one-line description is insufficient. Run `uv run reference/warp_api/warp_version.py` to compare every
stamp against the installed `warp-lang` and see which files an upgrade has left stale.
`reference/warp_api/REGENERATE.md` records how to re-extract them.

**A name in `dir(wp)` that is missing from those mirrors is usually hidden on purpose, not missed by
the transcription.** `dir(wp)` exposes ~510 names against ~137 the package uses, and browsing the
remainder for adoption candidates is how the `wp.dense_chol` / `dense_subs` / `dense_solve` family
gets proposed as a replacement for a hand-written 6×6 Cholesky. Introspect before planning around one:

```python
from warp._src.context import builtin_functions
f = builtin_functions["dense_chol"]      # a Function, the same handle §2.7's factories capture
print(f.hidden, f.doc, f.input_types)    # True  'WIP'  {n: int32, A: array(ndim=1, float32), ...}
```

`hidden: True` / `doc: "WIP"` is the answer, and the mirrors' silence was the same answer read one
step earlier.

**Then check the *quantity*, not the name — adopting a matching builtin is sometimes a regression.**
Also check the signature's *storage class* and its precision: a builtin that only speaks `float32`
cannot serve the `float64` half of a dispatch, so half the hand-written code stays either way. And
where a builtin *does* fit, the argument is often single-source-of-truth rather than speed —
`wp.volume_index_to_world` measured perf-neutral (1.08x at 200k voxels, 1.005x at 2M, both
launch-dominated) against a hand-rolled half-voxel transform that agrees with it to 3.58e-07; adopt
it for the convention, and if only half the sites can convert, **name the split in the module
docstring** rather than leaving two silent conventions in one file. The measured adoptions and
rejections are §12.8.

**When a plan proposes swapping in a Warp geometry builtin, sweep the *scale* axis and the degenerate
cases, not just random inputs at unit scale** — unit-scale random pairs showed perfect parity and
would have hidden every defect §12.8 records.

---

## 11. Running long commands: never poll with `until`

The benchmark suite and the measurement probes §9 asks for routinely run for minutes. **Do not write
a wait loop around them.** A backgrounded command re-invokes you when it exits, reporting its exit
code and its output file path. Launch it, do something else, and read the output file when the
notification arrives.

- **`until ! pgrep -f <name>; do sleep 5; done` never terminates.** `pgrep -f` matches the *full
  command line of every process*, including the polling shell itself, whose command line contains
  `<name>`. The condition is therefore permanently true. Measured cost of not knowing this: **seven
  such loops in one session, each spinning for 3-4 hours** until killed by hand, every one redundant
  because the command it was watching had already sent its completion notification.
- **A watcher is never the record.** The benchmark's or probe's own stdout file is. If you find
  yourself launching a second command to learn whether the first finished, delete it.
- **When a poll genuinely is unavoidable** — external state the harness cannot see — make the pattern
  unable to self-match (`pgrep -f '[p]robe_p4'`) or test a sentinel file the process writes on exit,
  and give the loop a **bounded** iteration count. **The bracket trick is not sufficient on its
  own**: this harness runs a command as `zsh -c '… && eval '\''<your command>'\''…'`, so the pattern
  string appears *literally* in the invoking shell's command line and `pgrep -f` matches that shell
  even though it does not match the pattern. Measured: `pgrep -af '[p]ytest'` returned exactly one
  row, its own wrapper. **Prefer the sentinel file, and read a `pgrep` hit's command line before
  believing the process it names is real.**
- **Do not chain short sleeps** to approximate a long wait. Pass a longer `timeout`, or background it.

A foreground command that outruns its timeout is *also* moved to the background and notified the same
way, so exceeding a timeout is not a reason to start polling either.

**Do not run a timing probe while `pytest` or `zensical` is running** — the same reconstruction read
527-752 ms under contention and 295-400 ms on a quiet box.

---
---

# PART II — MEASURED FACTS

Everything below was measured on this box (**RTX 5090**, 170 SMs) against the Warp version stamped at
each item. Read the relevant section before proposing an optimization, diagnosing a slowdown, or
re-deriving a number.

---

## 12. The Warp platform: bugs, quirks and version status

### 12.1 Memory safety — every failure here is silent

- **A `wp.launch` with `device=` omitted corrupts the host heap, not a style nit.** It silently
  resolves to the default device (`cuda:0` whenever CUDA is present) while the array arguments sit
  on the CPU, and returns numerically correct results (HMM lets the GPU legitimately dereference
  host memory) while corrupting the heap, because the launch is asynchronous and the host arrays get
  freed while the kernel is still reading them. **A `wp.synchronize()` after the launch is the fix**
  — the asymmetry is that a CUDA array's storage frees with the stream-ordered `cudaFreeAsync` while
  a CPU array's frees immediately and unordered, so the CPU side is the one that actually corrupts.
  Always forward `device=` from the input arrays (§2.1) so this configuration is never reached.
- **An out-of-bounds kernel write on the CPU device *is* glibc heap corruption**, because a CPU Warp
  array is host heap — deferring a range check to a downstream call is enough to trigger it.
  Range-check at the write, not later; `connected_component_parity_from_edges` still doesn't.
- **General lesson: neither of the above is "the Warp CPU backend corrupts the heap"**, the framing
  that survived months of subprocess isolation before these were found — both were application bugs.
  Two bisection techniques generalize past this case: `wp.config.mode = "debug"` compiles kernel-side
  bounds checks, but **a clean debug run is evidence about timing, not correctness** (enough work
  after a bad launch can let the kernel finish before teardown even with the bug present, giving a
  false-clean debug run); and a **component-swap bisection** — run a clean pipeline and substitute
  one library component at a time — finds a launch-ordering bug that line-level bisection can't.
- **A `wp.Mesh` with zero triangles silently corrupts CUDA allocator state.** The constructor
  succeeds, but the *next* unrelated CUDA allocation anywhere later in the process fails with a
  spurious OOM and cascades into `CUDA error 700`. Only `indices.shape[0] == 0` matters, not point
  count. Safe on `cpu`; still broken on CUDA. Never construct one on CUDA, **including in tests** —
  use a single-triangle mesh to reach an `n_faces < 2` guard. `triwarp.mesh.Trimesh.warp_mesh` would
  hit this for a zero-face mesh — a known latent issue, not fixed.
- **Python-scope gather silently ignores a non-contiguous index view's stride.** `payload[edges[:,
  0]]` returns the flattened buffer's leading entries instead of column 0, no exception, and a
  `wp.map` over the same view inherits the corruption while reading *faster* — so a benchmark alone
  reads as a win. `wp.copy` the index into a dense buffer first. Rule: §3.4.
- **`wp.copy(dst, src, count=0)` copies the *whole* source, and `wp.utils.array_cast` inherits it** —
  Warp's own back-compat rule reads `count == 0` as "not passed" rather than "nothing". A `count`
  derived from data (a compacted length, a readback) must be checked for zero at the call site and
  the zero case answered without calling `wp.copy` at all; same shape as §3.3's zero-length-slice
  raise and §3.7's capacity-versus-count rule — a legitimate empty case Warp spells as a special
  value rather than a length.
- **`wp.copy` into a *pinned* host buffer is a genuine async memcpy with no event, so reading it
  right after races and returns the *previous* value** — CUDA only blocks the host on a *pageable*
  destination. `triwarp._device.read_scalar` is the safe, device-split spelling; pinning buys
  nothing once done correctly. **It caches one scratch buffer per dtype, which is only safe when
  everything returned is a scalar** — for a vector/matrix dtype `.numpy()[0]` is a *view* onto the
  shared scratch, so two sequential reads alias and the first takes the second's value; the fix is
  to copy before returning. General lesson: a per-dtype cached buffer degrades silently the day
  someone passes a non-scalar dtype through it.
- **Warp's CPU work runs ~36x slower in a process where CUDA has been initialised** — it is CUDA's
  mere *presence* in the process, not the launch-access guard or GPU contention, and
  `CUDA_VISIBLE_DEVICES` must be set before the process starts. This is why both-device coverage is
  a two-process job (§7.2).

### 12.2 `wp.launch_tiled` runs one lane per block on the CPU

Still true on **Warp 1.17**. `wp.launch_tiled(kernel, dim=[...], block_dim=64)` executes **one thread
per block** on the CPU backend — `wp.tid()`'s lane index is always 0.

**The obvious probe says "fixed", and that is the trap.** `wp.tile_load` reads its whole tile out of
an array, is lane-independent, and was never affected. Only `wp.tile(x)`, built from *per-lane*
values, collapses on CPU. **Always probe the lane-constructed tile**, or you will conclude it is
fixed and delete a correctness branch. Two silent consequences, no exception either way: a block
reduction over a lane-constructed tile returns only the block leader's contribution, and even a
kernel with no tile intrinsics breaks if it relies on lanes covering the tile
(`idx = tile_i * TILE_1D + t` touches only every 64th element on CPU). Because
`tests/conftest.py`'s `device` fixture returns `cuda:0` whenever CUDA is available, this class of
bug is invisible to the default test run and needs an explicit `"cpu"` parametrize to catch (§7.2).

**The fix is one token: stride by `wp.block_dim()`** — it reads the launch's `block_dim` on CUDA and
**1** on CPU, costs nothing on CUDA, and makes the single CPU lane cover every element, after which a
tile reduction degenerates to a one-element tile that correctly returns that lane's own answer.
**It does not generalize to *partitioned* kernels** — a kernel computing `f = i * TILE_1D + t` at
`dim = n_blocks` has one lane per block dropping most of the range, and no stride change reaches it;
that needs the kernel to loop over its block's range, which is why some kernels keep a lane-free
`_sliced` sibling behind `_device.prefers_tiled_reduction`.

**The correctness boundary is narrower than it looks: only a partition stride that isn't
`wp.block_dim()` is wrong, on *either* device (§2.2) — a one-element CPU tile is harmless, and a
tile reduction works fine inside a `@wp.func` too.** A blanket comment forbidding "any tile reduction
on CPU" is itself a defect: it is over-broad and blocks real wins.

Open upstream issues, cited in `_device.prefers_tiled_reduction`'s docstring: **NVIDIA/warp#1480**
(CPU/GPU tile parity) and **NVIDIA/warp#1638** (efficient CPU block execution). The branch above
becomes removable only when CPU blocks run more than one logical thread.

**Warp exposes no grid-wide barrier** (no cooperative groups, no `__threadfence`), and a spin-wait
emulation over `atomic_*` needs all blocks co-resident and a fence Warp cannot spell. That is the
ceiling on every level-synchronous rewrite (§14.9).

**Warp's hash grid has no per-cell entry point.** `wp.HashGrid` exposes only `hash_grid_query` /
`hash_grid_query_next` (a sequential per-thread iterator) and `hash_grid_point_id`, so the cell
*walk* cannot be split across lanes, only the per-candidate arithmetic can — the walk is 70-73% of
the per-candidate cost, so a cooperative search that keeps `wp.HashGrid` is capped at ~1.37x. **Never
propose a warp-per-edge search that keeps the hash grid.** Warp *does* ship a block-cooperative BVH
walk (`tile_bvh_query_aabb` / `tile_query_valid` / `tile_bvh_query_next`) — §14.2 — and it has a
silent correctness bug, next.

**`wp.tile_bvh_query_aabb` returns out-of-range primitive indices once a traversal round overruns
its result buffer, and every caller must bound-check what it hands back.** A round appends each hit
with an unconditional atomic counter increment but guards only the *write* against a fixed
`result_buffer_capacity` (`WP_TILE_BLOCK_DIM * 5`, i.e. 160 at `block_dim=32`) — once the counter
runs past capacity, a lane reads uninitialised shared memory as a primitive index. Nothing raises,
and the `>= 0` test every documented example uses does not catch it, since a garbage word is
positive about half the time. **A query box grown by a distance bound is what reaches the overrun**
— the round count scales with how many primitives one box meets, so a big mesh plus a generous box
is the trigger and a tight box on a small mesh never gets near it, which is why this had gone years
unseen. Confirmed via `compute-sanitizer --tool memcheck` on
`proximity.mesh_to_mesh_distance`'s tiled straggler pass.

Two consequences:

- **Bound-check the index at every `tile_bvh_query_next` site**, `candidate >= 0 and candidate < n`.
  Both of triwarp's do. It is output-neutral by construction — an out-of-range index is never a
  primitive of the BVH — so it can only reject what was already garbage.
- **The guard stops the corruption and cannot restore the dropped primitives.** An overrun round
  silently loses hits, so a guarded walk may return an incomplete candidate set; that half is
  upstream's. In practice the two triwarp callers survive it because each has a *second*, sound
  bound on the answer — the global running minimum, and the pivot's own acceptance test.

**Warp exposes no node-by-node BVH traversal** either (`bvh_query_aabb` / `bvh_query_ray` /
`bvh_query_sphere` / `bvh_get_group_root` only), so a BVH-pair wavefront means writing our own
hierarchy — price it as that, not as a rewrite of the query.

**A `wp.capture_while` loop body that issues several launches does not replay as one unit on both
devices**, so a loop whose *result buffer* depends on the whole body running cannot rely on it.
Measured while removing a per-pass `wp.copy` from `graph.shortest_path_envelope`, whose
Bellman-Ford relaxation reads and writes every label and so needs a second buffer. Unrolling **two**
passes into the body, each writing into the other's buffer, removes the copy entirely and halves the
conditional-graph evaluations — worth 1.26-1.45x — but the CPU device relaxed markedly fewer nodes
per round than CUDA at every cap: it records through `ScopedCapture`'s APIC recording
(`wp.is_conditional_graph_supported()` is a *machine* query and returns `True` even for a CPU-device
array) and the body did not replay as two passes per round. Reverted. **The generalisable rule: a
Python-level ping-pong cannot help a captured loop either** — the body is recorded once and
replayed, so rebinding the names only takes effect at record time. Between those two, a
double-buffered iteration inside `capture_while` keeps its copy.

### 12.3 `wp.ref[T]` requires concrete types

Verified by compile probes; still broken on 1.16.

- **Generic type-vars do not instantiate inside `wp.ref[...]`** — `wp.ref[wp.Scalar]`,
  `wp.ref[wp.Int]` and `wp.ref[typing.Any]` all fail resolution *even for float32* ("Couldn't find
  function overload"). `wp.Any` does not exist. Concrete `wp.ref[wp.float32]` / `wp.ref[wp.int32]`
  work.
- **No `@wp.func` name-overloading**: a second `def foo` just shadows at Python scope, and
  `wp.overload()` is **kernels-only**. So one helper name cannot serve both float32 and float64.
- **Tuple-assignment swap** (`a, b = b, a`) works only for **simple local variables** (like `sort3`),
  NOT for array/matrix/vector elements — `arr[i], arr[j] = ...` raises *"Multiple return functions
  can only assign to simple variables"*.

Consequence: the shared argmin/argmax/swap helpers in `kernels/array.py` are concrete `float32`
(`update_argmin`, `update_argmax`, `update_argmax_lowest_index`, `update_argmax_vec3`,
`update_argmin_pair`). Float64 sites and int32 array-element swaps keep hand-written loops.

### 12.4 Numerics and precision

- **`wp.Float` covers `float16` / `float32` / `float64` from one generic definition** — that is what
  let `array.allclose` drop its hardcoded `float32` path. **`wp.Scalar` does NOT instantiate for
  `wp.bool`** (`TypeError: Function <name> does not support the provided argument types
  warp._src.types.bool`; it does work for `int8` / `int32` / `float32`), so a "works on masks and on
  numbers" wrapper needs a dtype branch, not one generic func.
  - **Do not answer that branch by widening the mask to `int32` first — write the concrete bool
    kernel.** `astype(mask, wp.int32)` is an allocation of `4n` bytes, a launch, a full read of `n`
    and a full write of `4n`, after which the reduction reads `4n` rather than `n`: **nine bytes of
    traffic per mask byte** to answer one boolean, and measured ~2x on the whole call. There is no
    `wp.tile_load` of a `bool` array, so such a kernel cannot take the tile-load shape: its lanes
    walk the block's chunk striding by `wp.block_dim()` (§2.2, which is also what keeps it right on
    the CPU device) and the block folds the per-lane registers with a single `wp.tile` reduction.
    The `counts_to_offsets(astype(mask, wp.int32))` sites are *not* the same defect —
    `wp.utils.array_scan` genuinely cannot read a `wp.bool` buffer — but they are no longer an
    `array_cast` either: `kernels/array.bool_flags` writes the same 0/1 bytes at half the cost, flat
    in the array size.
  - **It moves a documented crossover, so re-check the declines that cite one.** The
    readback-versus-device-reduction crossover for a `wp.bool` mask halved, to ~0.5 M elements.
    Both sites in `repair.py` carrying a decline of that shape still sit under it and keep their
    readback — with the new number written in place of the old one, which is §9's discipline, not a
    conversion.
- **Scalar arguments must be constructed at the input's precision**: `wp.map(f, a_float64,
  wp.float32(tol))` will not resolve the generic — use `a.dtype(tol)`. Literals *inside* a generic
  `@wp.func` need `type(x)(...)`.
- **There is no `wp.any` / `wp.all` over vector components** and no generic vector annotation beyond
  `Any`, so per-component predicates stay as named per-type funcs behind a dtype-keyed dispatch
  (`is_close_vec3`).
- **FMA fusion makes a degenerate triangle's area ~1e-8 on CUDA and exactly 0 on CPU.**
  `triangle_cross` computes `wp.cross(v1-v0, v2-v0)`; for a face with a repeated vertex the two edges
  are numerically identical, so the cross is exactly `0` on CPU but ~`1e-8` on CUDA, because Warp's
  `fuse_fp` is ON by default and fuses `a*b - c*d` into `fma(a,b,-(c*d))`. That flips a zero-area
  face to "nondegenerate" against the absolute `TOLERANCE_MERGE = 1e-8` altitude test at unit scale.
  **Decision: leave the imprecision — do NOT disable `fuse_fp`.** Disabling it would kill FMA fusion
  across the whole triangles kernel module (closest_point, barycentric, normals — proximity/sampling
  hot paths), and because `@wp.func`s compile under each *calling* kernel's module options it would
  have to be disabled in every module using `triangle_cross`. **So write degeneracy tests with
  scale-aware inputs** — scale vertices to ~1e-2 so genuine altitudes stay far above `1e-8` while the
  FMA residual stays far below it — and never assume CPU/GPU bit-agreement on zero-area faces.
- **The same contraction makes `-orient2d(p)` and `orient2d(mirror_y(p))` different predicates
  on CUDA.** Algebraically they are one value; under `fuse_fp` the two expressions contract
  differently, and a near-collinear vertex flips its convex/reflex verdict. Found folding
  `polyline_triangulate`'s reflex count for the y-mirrored loop into the turning-angle pass: the
  negated form pushed a 20 000-point convex ring off the fan path on CUDA only. The kernel
  evaluates the mirrored loop explicitly (`kernels/polyline.mirror_y`). **Never substitute an
  algebraic identity inside a sign test that feeds a branch** — it is bit-exact on CPU and not on
  CUDA, so a CPU byte-identity gate will not see it.
- **`wp.mesh_query_point_no_sign` + `wp.mesh_eval_position` is not an exact closest-point query** —
  it disagrees with an independent float64 oracle by up to ~2e-5 absolute, always reporting the
  *smaller* distance, and is a fixed point (re-querying from its own answer doesn't converge closer).
  A distance *bound* built on it is therefore exact only against Warp's own query
  (`proximity.closest_point_on_mesh`), not against an independent oracle. **How to test such a
  bound:** assert exactness against the same Warp query the bound was built from, and assert an
  *improvement ratio* against an independent oracle rather than the bound value itself — asserting
  `independent_distance <= bound` fails at tight bounds for a reason that isn't a bug.
- **`wp.length(d) < r` and `wp.length_sq(d) < r*r` are not the same predicate in float32** — 10 rows
  of 200k disagree at the boundary, the same 10 on CPU and CUDA. Rule: §2.4.
- **`NaN` breaks a binary search, and which way it breaks depends on the search's convention.**
  Every comparison against `NaN` is false, so a midpoint landing on one is steered by a predicate
  that never fires. `searchsorted(side="right")`'s `values[mid] > value` then advances `left`, i.e.
  *into* Warp's float radix sort's trailing `NaN` block, and the search returns the array length for
  every finite value; `side="left"`'s `wp.lower_bound` is steered away from it and is correct for
  every finite value while landing `NaN` itself at slot 0. Neither raises. Where a `NaN` can only
  belong to the last slot — which is exactly a sorted-unique table — the fix is the left search plus
  `if index >= n or values[index] != value: index = n - 1`, since `NaN` compares unequal to
  everything including itself. §16.5 has the case this cost.
- **float32 storage sets a hard noise floor that no solver tolerance reaches.**
  `tangent_space.halfedge_transport_angles` (and its dependents) are float32, and that storage
  precision — not the CG tolerance, not `vertex_tangent_frames` (which only names the reporting
  basis and isn't read by `connection_laplacian` at all) — sets the noise floor of everything built
  on `laplacian.connection_laplacian`. **The noise floor overlaps real signal**, so no magnitude
  threshold separates them; report it with a resolved/unresolved mask instead of a cutoff, which is
  why `heat.transport_tangent_vectors` returns `(transported, resolved)`.

  **General technique for "is this solver noise or input precision?", cheapest first:** (1) sweep the
  solver tolerance — flat means it isn't convergence; (2) inject noise into the suspected input and
  check linearity, extrapolating back to the unperturbed point; (3) split storage from arithmetic —
  redo in float64 but round the result back to the shipped dtype; unchanged means it's the *storage*,
  and an internal-only fix can't work.
- **Diffused heat fields scale as 1/scale² with the mesh coordinates**, so an absolute tolerance
  applied to one is a silent wrong answer on a rescaled mesh.
- **The backward pass of an *empty* dynamic `range` is not empty (Warp 1.17), so a differentiated
  thread whose strided loop draws nothing must return before the loop.** The adjoint walks
  `iter_reverse(range(start, end, step))` (`warp/native/range.h`), which sets
  `start + int((end - start - 1) / step) * step` -- C++ truncation, so for `start >= end` with
  `end - start - 1 > -step` it is one iteration at `start`, past the end of the array. The forward
  is untouched, so the loss is right and only the gradient is wrong, by an out-of-bounds read and
  scatter: 11x the true gradient on `metrics.chamfer_nn_term_sliced` at 1 point over 37 slices,
  and a heap-corruption crash on the CPU device. Both sliced chamfer kernels now open with
  `if j >= n: return`, pinned by `test_sliced_chamfer_terms_grad_with_more_slices_than_points`.
  `slice_count` never launches more slices than elements, so no public call reached it; a probe or
  a test choosing its own slice count did. Only `metrics` is taped, so no other kernel is exposed.

### 12.5 Kernel-scope cast semantics

All measured on Warp 1.16 (probe scripts written to a file — Warp refuses `exec()`-defined kernels).

- **`int(x)` and `wp.int32(x)` are the same operation.** Generated C++ is identical apart from the
  call name (`wp::int32(var_0)` versus `wp::int(var_0)`), and the latter only compiles because Warp
  writes an **unconditional** `#define int(x) cast_int(x)` / `#define float(x) cast_float(x)` into
  every module header (`warp._src.codegen.cpu_module_header`) — `wp::int(x)` is not valid C++. There is no cost to either spelling. The one
  recorded difference is in Warp's own type system: `int`'s `value_type` is Python `int` and
  `float`'s is Python `float`, where `int32`/`float32` carry the Warp types.
- **That difference bites exactly once, and it is fatal.** `total / float(count)` inside a
  `wp.Float`-generic `@wp.func` does **not** silently narrow — it fails to parse: `Input types must
  be the same, got ['float64', 'float32']`. So every `float()` in a kernel is a hard commitment that
  its enclosing function will never be made generic.
- **No cast on a thread index is load-bearing.** A bare `wp.tid()` passes unchanged to a `wp.int32`
  `@wp.func` parameter, to a `wp.Scalar`-generic parameter, and into a kernel-scope slice; all
  spellings compile and agree.
- **`//` is not CPython's `//`** — §1.5 has the table.
- **None of this describes *Python* scope, where the same spellings behave oppositely.** In a kernel
  `wp.int32(x)` is a cast that costs nothing; at Python scope it is a *constructor* for a
  `warp._src.types.int32`, whose arithmetic then routes through Warp's builtin dispatch at ~10 µs an
  operation (§13.1). Three consequences worth carrying across the boundary: `//` and `%` on such a
  value **raise `TypeError`** rather than truncating; `wp.float32(x)` does **not** round, since
  `scalar_base.__init__` is `self.value = x`; and `int(x)` / `float(x)` are the *unwrap*, costing
  ~0.08 µs, which is the fix rather than the defect. Check 26 (§4.5) scans for it.

Rules: §1.3, §1.5, §1.6.

### 12.6 Compilation, module hashing and import cost

- **A generic kernel's lazy overload instantiation rebuilds its whole module** — §2.5 carries the
  measurement (206 → 87 module loads, suite 1 033 s → 29 s).
- **`wp.map` forks its generated module per call *signature*, on axes wider than the dtype** — §3.5
  carries the measurement (182 → 143 loads; cold cache 2.1x).
- **`import triwarp` was expensive because `@wp.kernel` builds an `Adjoint` at import time for every
  decorated kernel, and importing one submodule doesn't avoid it** (Python imports the parent package
  first). Fixed with a PEP 562 module `__getattr__` that resolves each submodule lazily — details and
  the guarding test are in §16.2.
- **A deferral one module makes can be silently cancelled by an unrelated module's top-level
  import — check `sys.modules`, don't trust the comment.** `triwarp/reconstruction.py` defers
  `import warp.fem` on purpose, but two *kernel* modules were loading the whole package eagerly for
  two `@wp.func`s, so the deferral saved nothing. `warp.fem`'s public re-export shim is far more
  expensive to import than `warp._src.fem.linalg` directly (it pulls in `adaptivity`, `dirichlet`,
  `domain`, `field.*`, `geometry.*`), so the tree imports the private module instead, with a comment
  naming the Warp version this was probed against — the risk is an upgrade moving `warp._src`, which
  fails loudly at import rather than silently.
- **Only host readbacks block a CUDA graph capture.** `array_scan`, `bsr_from_triplets`,
  `radix_sort_pairs` and a nested `wp.capture_while` all capture and replay fine on Warp 1.16.
  `edges_unique` / `flatnonzero` / `remove_unreferenced_vertices` each read back the count that sizes
  their own output — that raises `CUDA error 906` inside a capture — and each was replaced by the
  inclusive scan it was reading, with the count left in a device `state` array. **`warp.Graph`
  retains modules but not arrays**: keep a reference list for anything allocated inside a capture
  (probing could not reproduce corruption after dropping it, so this is defensive, not proven).
- **`block_dim=32` takes the `warp_count == 1` fast path** in `tile_reduce_impl` (a ballot plus a warp
  shuffle, no cross-warp shared round trip) — that is the whole 126-vs-369 ns gap in §13.2.
- **"Clear the kernel cache before the first run" is a no-op across an upgrade**: Warp namespaces the
  cache by version (`~/.cache/warp/1.17.0`), so a new version is cold by construction. Also
  `wp.config.kernel_cache_dir` reads `None` until init — the real path only appears in the init banner.
- **`wp.config.verbose = True` is deprecated in Warp 1.17** — it prints a deprecation notice to
  stderr, which is noise in exactly the output you are grepping. Use
  `wp.config.log_level = wp.LOG_DEBUG`; the log lines themselves are unchanged.
- **coverage.py cannot see a kernel body, so `triwarp/kernels/` is omitted from the coverage
  measurement.** Same root cause as the import-cost bullet above: `@wp.kernel` / `@wp.func` parse
  the function's AST and codegen C++, and the Python function object is never called, so the
  tracer records a kernel that runs on every test as entirely unexecuted. Measured on
  `tests/test_edges.py` (53 tests, CPU device): the wrapper `triwarp/edges.py` reports **96 %**
  against `triwarp/kernels/edges.py`'s **35 %**, and the missing ranges are exactly the
  `@wp.func` and `@wp.kernel` bodies — `_write_edge`'s lines 11-16, `faces_to_edges`' 24-31 —
  every one of which ran on all 53. The figure is not a pessimistic reading of kernel testing, it
  is unrelated to it. Tree-wide over the whole CPU suite (3 397 passed, 28 skipped, Warp 1.17), in
  *line* coverage so the two halves compare directly: the wrapper layer is **96.02 %**
  (10 949 / 11 403 statements) against `kernels/`'s **24.29 %** (2 890 / 11 896), so *including*
  kernels would publish **59.40 %** — a number that measures nothing and would invite exactly the
  wrong work. With branch coverage on, which is what ships, the wrapper layer reads **93.79 %**
  and that is the figure CI's `coverage report --fail-under=90` gates — a deliberately loose
  floor, ~3.8 points of slack, set as a regression alarm rather than a target. The gate is a
  *separate step after* the badge publish, so a regression on `main` still moves the badge before
  it fails the build; gating inside the pytest step would freeze the badge at the last good
  number.
- **A second coverage artifact, from the same root cause one level out: `wp.map` leaves
  unparseable filenames.** `warp._src.utils.map` builds its generated kernel with `exec` and names
  the module after the *call site* — `warp/_src/context.py` spells it `f"{basename}:{lineno}"` —
  so a code object turns up whose filename is `triwarp/points.py:153`. coverage.py tries to read
  that as a path and emits one `couldnt-parse` warning per generated module: **72** over a full
  suite run, contributing no lines to the report. Omitting `*.py:*` removes all 72 and changes no
  count (11 403 / 454 / 3 320 either way), since no real source file has a colon in its name. It
  only reproduces on a *whole-suite* run — three single-file runs produced zero — so do not try to
  reproduce it on one test module. Configured in `pyproject.toml`; the rule is §8.
- **A `@wp.func` is not inlined at codegen — Warp emits a real `static CUDA_CALLABLE` function and
  calls it, and the inlining is nvcc's.** The tree says "inlined at codegen, so this costs nothing"
  in several places (§2.4 among them) and the *conclusion* holds — every extraction measured here
  compiled to byte-identical or smaller SASS — but the mechanism is the compiler's, not Warp's, so
  it is a claim to verify rather than a guarantee to assume. A tuple return additionally emits a
  `wp::copy` per component into the out-parameters, which nvcc also elides.
- **Proving a `@wp.func` extraction is cost-neutral is free, needs no clock, and is the right tool
  on a busy box (§15.6).** Warp caches the generated `.cu` *and* the compiled `.ptx` per module
  under `~/.cache/warp/<version>/wp_<module>_<hash>/`, so the whole recipe is: load the module on
  `cuda:0` in each arm (the new tree, and a detached worktree — §15.6), read the module hash off
  `list(warp._src.context.get_module(name).hashers.values())[0].get_hash()`, pick that hash's cache
  directory, and count instructions per `.visible .entry` in the PTX. Strip the per-kernel
  `_<8 hex>_cuda_kernel_` infix or the changed kernels will not match across arms, and read only
  the `_forward` entries.
    - **Go one step further to SASS, because ptxas folds most of what a PTX diff reports.** Four
      kernels that moved in PTX (−18, −8, +1, −1) were byte-identical in SASS; two real changes
      survived. `ptxas -arch=sm_120 -O3` plus `nvdisasm -c` is the whole pipeline, out of
      `/usr/local/cuda-12.8/bin` (not on `PATH`). That toolchain's ptxas is one PTX version behind
      Warp's bundled NVRTC, which fails as `Unsupported .version 8.8`; rewriting the `.version`
      line to `8.7` assembles fine for this instruction set.
    - **Count `nvdisasm -c` lines on an address regex of `/*[0-9a-f]{4,}*/`, not `{4}`, or the
      count silently saturates at 4 096.** SASS addresses are 16 bytes apart, so the comment
      switches to five hex digits at `0x10000` and a four-digit pattern stops matching there. The
      tell is two *different* kernels reporting exactly 4 096 — which is what a
      `face_to_mesh_distance` / `_tiled` pair did, hiding their real 10 512 and 11 448 behind a
      false tie. A truncating counter fails toward "identical", i.e. toward the answer an
      extraction wants to hear.
    - **Compare the `_forward` entries and expect the `_backward` ones to move.** A `@wp.func`
      extraction that is exactly free forward still reshuffles the generated adjoint: across five
      modules, 103 of 103 forward entries were identical and four backward ones shifted by −320 to
      +32. That is not evidence of a cost, because nothing outside `kernels/metrics.py` is
      differentiated (§2.6) — but a diff that reads every entry reports it as one.
- **Warp lowers a kernel-scope `not` on a bool as a *select*, not by flipping the comparison, so a
  boolean `@wp.func` imposes a polarity on its callers and the complement costs a select per call.**
  Measured on `voxels.count_box_faces`, whose six-row stencil loop unrolls: sharing the neighbour
  probe as `-> wp.bool` and writing `if not probe(...)` cost **48 SASS instructions**, where sharing
  it as `-> wp.int32` (the grid slot) and leaving each caller its own `< 0` / `>= 0` is exactly
  free. Counting the complement and subtracting (`6 - covered`) was tried and is *worse* (+88).
  **So a shared predicate over a sentinel returns the sentinel, not a boolean**, which is what
  `voxels.cell_slot` / `point_slot` already did and the reason to follow them.
- **`wp.constant(x)` is `return x` after an `is_value(x)` check — on Warp 1.17 it is an identity
  function, not a declaration.** A bare module-level global with no `wp.constant()` and no typed
  constructor compiles and runs correctly from kernel scope on both devices, because Warp's codegen
  resolves *any* free variable that evaluates to a static value, wrapped or not — so
  `wp.constant()` adds no kernel-scope visibility, and `triwarp/constants.py` no longer calls it
  (§1.2). It also fixes no dtype. What it still legitimately does: raise `TypeError` immediately at
  the definition site if the value isn't scalar/vector/matrix-shaped, rather than failing later and
  more confusingly inside a kernel that references it.
- **A tile `shape=` must be a plain integer, so `wp.constant(wp.int32(n))` cannot serve as one.**
  `wp.constant(256)` works as a `wp.tile_load` / `wp.tile_zeros` `shape=`; the typed spelling fails
  at *parse* time with an `AttributeError` naming the kernel. That matters because §1.5's check 17
  types an operand by declaration and only recognises an integer `wp.constant` in the typed form, so
  a constant that is also a tile shape can never be made visible to it — `algorithms/bfs.py`'s
  `BFS_SCAN_BLOCK` converted (it is only an offset and an index). `conjugate_gradient.py`'s
  `CG_TILE` was the standing counter-example until its one tile-shape use, a `tile_load`
  accumulator in `cg_seed`'s fold, became a lane-strided `reduce.block_sum`; it is now a plain host
  integer, since no kernel reads it. A typed constant is also not usable in *host* arithmetic
  (`(n + c - 1) // c` raises `unsupported operand type(s) for //`), so the wrapper reads `int(...)`.

### 12.7 `warp.sparse`

The `nnz`-is-a-capacity rule and its consequences are §3.7. Three further behaviours, all silent:

- **`bsr_mm` returns a structural *superset* of the product**, with the extra entries exactly zero
  but interspersed in column order rather than trailing capacity (not NVIDIA/warp#1769, which is
  about trailing capacity — this is a different, still-present behavior on 1.17). **The zeros are
  not cosmetic once the result is multiplied again**: an explicit zero at `(i, c)` makes column `c`
  "see" row `i`, so a Galerkin product `PᵀAP` inherits every aggregate reachable from it — a measured
  47x pattern blowup on one coarse operator, operator complexity 2.19 instead of a true 1.03.
- **`bsr_compress`'s illegal-memory-access is FIXED on Warp 1.17** (was: compressing a `bsr_mm`
  result made the *next* `bsr_mm` die with `CUDA error 700`). `linalg._multigrid_prune` is now one
  call to `wps.bsr_compress(matrix, prune_numerical_zeros=True)`, ~2.9x faster than the CSR-to-triplets
  rebuild it replaced. **Trap found doing it: at the documented default `inplace=False`, it still
  returns the *same* matrix, pruned in place** — copy first if the unpruned matrix is still needed.
- **`bsr_set_transpose`, `bsr_mm` and `bsr_axpy` all read the `nnz` *field*, never `nnz_sync()`**, so
  a matrix whose count is stale carries garbage into whatever consumes it — the operand-side version
  of §3.7's buffer-sizing rule.
- **`bsr_mv` takes one vector**, so a cycle over several right-hand sides pays one launch per vector
  per mat-vec; a hand-written batched CSR kernel amortizes that across columns (shipped in the
  multigrid V-cycle).
- **Unwritten `(0, 0, 0.0)` triplets in a `wp.zeros` buffer all accumulate on entry `(0, 0)`, and
  `bsr_from_triplets`'s duplicate accumulation costs O(duplicates on the hottest address), not
  O(triplets).** A conditional triplet writer must point its unwritten slots *out of range*
  (`rows.fill_(n_rows)`, silently dropped) rather than leave them at a zeroed default — measured
  9-32x depending on how many triplets collide on `(0, 0)`. §3.7 is right about correctness (a
  structural zero is harmless) and wrong about cost; price a conditional emit by its collision
  count, not its buffer size.
- **`warp.optim.linear`'s `TiledDot` silently picks an O(n) per-block reduction whenever
  `batch_offsets` is set with `batch_count > 1`**, instead of its normal O(log n) tiled-tree
  reduction — and `use_bounded_tree`, the escape hatch, requires `batch_count == 1`, so a batched
  solve can never reach it. Two dots run per CG iteration, and at moderate `n` this can dominate the
  iteration cost. `linalg._BatchedCg` works around it with a real two-stage per-column tree
  (1.10-1.50x end to end, iteration counts unchanged); running one unbatched `cg` per column instead
  reaches the good reduction but duplicates every other kernel in the iteration and loses overall.
- **`warp.optim.linear.cg` at `check_every=0` records, instantiates and launches a fresh
  conditional graph on every call** (`_run_capturable_loop` opens a `wp.ScopedCapture` around its
  `capture_while` each time), so every single-column `solve_spd` pays §14.3's "record, replay
  once" row. Measured on a heat solve at 2 562 and 10 242 vertices, flat in the size: **2.07-2.10 ms**
  through `wpl.cg`, **1.33-1.36 ms** through a fresh `_BatchedCg` (which records too), **0.61-0.63 ms**
  replaying a persistent `_BatchedCg` (`plans/benchmark-round-16-data/probes/solve_reuse.py`). On
  small and medium systems the graph costs more than the iterations; keep the solver object where
  the operator is reused. **Addressed in round 16** (§16.16): no scalar or `wp.mat22d` solve in the tree
  reaches `wpl.cg` any more, and a solve keeps one recorded state per operator.
- **`warp.optim.linear.cg` silently resolves an omitted `atol` to `atol := tol`, turning a relative
  residual tolerance into an absolute floor of the same numeric value** — its convergence criterion
  is `max(atol, tol * ‖b‖)`. Once a right-hand side's norm falls below that floor, the solve returns
  **zero iterations** and the untouched initial guess as "converged" — a correct-*looking* silent
  failure, not a flagged one. **FIXED**: every triwarp call site now passes `atol=0.0` explicitly
  alongside `tol=` (`linalg.solve_spd`, `linalg.solve_spd_columns`'s single-column path, and both of
  `reconstruction`'s direct `wpl.cg` calls) — triwarp's own multi-column path (`_BatchedCg`)
  was never affected, since it computes its stopping threshold with an explicit
  `atol_sq = 0.0` already. This is what let a mesh-scale-dependent right-hand side (as in
  `heat.log_map` at extreme mesh scale) silently return zero instead of solving. Verified free at
  ordinary mesh scale: iteration counts and benchmark timings are unchanged before/after on normal
  inputs, confirming the fix only changes behavior once `‖b‖` is already pathologically small.

### 12.8 Warp builtins: adoption verdicts

**Adopted:**

- **`wp.mesh_query_point_sign_winding_number`** → `proximity.signed_distance_on_mesh(...,
  sign_mode="winding")`. Exact where ray parity misclassifies on meshes with holes, at a memory and
  runtime cost, so parity stays the default. Two traps: `support_winding_number=True` is required or
  the builtin **silently returns the ray parity answer**, and `wp.Mesh` doesn't retain the flag, so
  it can't be checked from Python — only a function that builds its own mesh can guarantee it.
  Warp exposes only the thresholded *sign*, not the value, so `proximity.winding_number` still needs
  a custom LBVH.
- **`wp.bvh_query_sphere`** (Warp 1.17), adopted in `kernels/neighbors.py` and
  `proximity.py::closest_point_on_edges` — a real win where the enumeration radius is large relative
  to an existing BVH. **It is bit-exactly `wp.length_sq(d) <= r*r`, not `wp.length(d) <= r`**, so
  adopting it means adopting the squared predicate everywhere, including the hash-grid branch.
  **Do not convert a probe whose radius equals the hash-grid cell width** — that's the grid's own
  best case, and the identical conversion in `ball_pivoting.ball_is_empty` was reverted as a real
  loss there (a small, well-centred query reaches the hash grid's target cell by address arithmetic,
  where a BVH always pays a root-to-leaf descent).
- **`wp.bvh_query_sphere` again, as a broad phase over *bounds*.** Adopting it in
  `curvature.discrete_mean_curvature` (in place of a cube `wp.bvh_query_aabb`) won 2-4x on the whole
  public call. **The platform fact underneath it: `wp.bvh_query_aabb`'s traversal costs several times
  `wp.bvh_query_sphere`'s per candidate returned on the same BVH, even when made to return the same
  candidate count** — so a cube broad phase is the wrong query whenever the caller's real predicate
  is a ball, independent of the extra candidates it returns. Not yet applied to `ball_pivoting`'s
  pivot search, which uses the tiled BVH walk and has no `tile_bvh_query_sphere` counterpart in 1.17
  — an open lead, unmeasured.
- **`wp.mesh_get_bvh`** (Warp 1.17) — `proximity.mesh_to_mesh_distance` now builds one structure over
  mesh B instead of two; §16.6.
- **`wp.volume_index_to_world`** — adopted for the convention, not the speed (perf-neutral against
  the hand-rolled transform it replaced).

**Rejected on measured evidence — do not re-propose without new data:**

- **`wp.intersect_tri_tri` cannot replace `intersection.triangles_intersect_sat`.** Its epsilon is
  absolute on unnormalized plane distances, so its verdict moves with mesh scale where triwarp's SAT
  is scale-invariant — it disagrees badly at both very small and very large scale, in either
  precision, and `mesh_with_mesh` is public API on arbitrary meshes.
- **`wp.closest_point_edge_edge` cannot replace `remesh._segments_dist_sq_d`.** Float32-only (the
  float64 branch of the caller never executes against it), and measurably worse relative error on
  near-parallel segments — exactly the near-degenerate case the existing float64 port exists for.
- **`wp.sample_unit_hemisphere_surface`** would replace `visibility`'s low-discrepancy Fibonacci
  lattice with a Monte-Carlo estimate at the same ray count — variance where there was none.
- **`wp.norm_huber`** is the Huber *norm* where `registration.robust_weight` needs the IRLS *weight*
  `ρ'(r)/r`.
- **`wp.tile_arange`** cannot express `array.arange` and loses where it can — its bounds are read at
  codegen, so a runtime start is a parse error, and a range fill has no reuse for a tile to exploit
  anyway (§13.1 already prices that call as launch- and allocation-bound).
- **`wp.volume_voxel_count`** is a capacity (§3.7).
- **The `dense_chol` / `dense_subs` / `dense_solve` family** is `hidden: True` / `doc: "WIP"`, and it
  takes `wp.array[float32]` where the caller holds a `wp.spatial_matrix` in registers — a 2x loss
  (§2.9).

### 12.9 `wp.Volume` as a voxel-set container

`triwarp/voxels.py` ships all of this and its module docstring carries the detail. On Warp 1.16+,
`allocate_by_voxels` and `fem.Nanogrid` both work on **CPU** with identical counts and byte-identical
`get_voxels()` row order across devices, so a volume-backed module doesn't have to be CUDA-only; an
empty point set raises `RuntimeError` rather than aborting the process, but still guard it.

- **`Volume.allocate_by_voxels(world_points, voxel_size, translation)` deduplicates**, and beats a
  hand-rolled `unique_rows` dedup by ~2.3x on both devices.
- **`volume_lookup_index(grid, i, j, k)` is the `k`-th row of `get_voxels()`**, and `-1` when absent
  — the grid and the cell array share one canonical numbering, so a per-voxel payload is just a
  `wp.array(n_voxels)`. No side table, no hash map, O(1) membership.
- **NanoVDB centres voxels on integers**, so to make its index equal a corner-aligned cell index
  (`floor((p-origin)/s)`), pass `translation = origin + 0.5*voxel_size` — without the shift every
  cell comes back +1 on every axis, silently.
- **`get_voxels()` order is leaf-major**, deterministic but *not* globally lexicographic — lexsort
  before comparing to a reference and never assume it matches a C-order reshape.
- **`voxel_points` may be integer** — a contiguous `(n,3)` int32 (or `vec3i`) array is read directly
  as index-space cells, no float round trip.
- **`point_mask` (int32, one per point, 0 = ignore) filters during the build**, so no compaction
  pass is ever needed, and a fully-masked one-point build is the only way to make a legal EMPTY
  volume (a zero-length input raises instead).
- **Rebuildable volumes: the payoff isn't there** — `rebuild()` into a pre-reserved topology is only
  marginally faster than a fresh `allocate_by_voxels`, so it doesn't pay for `dilate`-style loops.
  Two traps if used anyway: `get_voxel_count()` returns the reserved *capacity*, not the active count
  (read `get_active_stats().voxel_count`, §3.7); and the four `max_*` capacities **cascade** — passing
  only `max_active_voxels` under-reserves the other three, runs out of device memory, and leaves the
  CUDA context throwing illegal-memory-access on everything after. Pass all four explicitly.
- **`warp.fem.Nanogrid(volume)` derives topology triwarp would otherwise hand-write** — verified
  exact against a hand-built solid-block topology. `.vertex_grid` is the deduplicated corner lattice;
  `boundary_side_index()` + `side_position` + `side_normal` enumerate outward faces;
  `side_inner_cell_index` over boundary sides is the 6-connected surface-voxel set. Import `warp.fem`
  **inside the function**, never at module scope (§12.6).

### 12.10 Upgrade discipline and the workaround table

Status of every version-stamped workaround, last full re-probe against 1.16 with 1.17 deltas noted:

| Workaround | Status |
|---|---|
| `warp.optim.linear.cg` returns NaN on CPU | **FIXED in 1.16** — cpu and cuda:0 identical to 1.7e-10 on a 64x64 SPD system. `_device.require_cuda` and all seventeen call sites deleted |
| CPU backend corrupts the process heap | **FIXED / never was Warp** — see §12.1; the repro that aborted ~60 % on 1.15 ran clean 12/12 |
| `bsr_mm` nondeterministic on a chained triple product | **REFUTED — never a Warp bug** (§3.7) |
| `bsr_compress` illegal memory access (#1769) | **FIXED in 1.17** (§12.7) |
| `bsr_mm` structural superset | **still present on 1.17**, and it is not #1769 (§12.7) |
| `wp.launch_tiled` one lane per block on CPU | **still broken on 1.17** (§12.2) |
| empty `wp.Mesh` corrupts the CUDA allocator | **still broken** — 10/10 subprocesses died on the next 4 MiB alloc (§12.1) |
| `wp.ref[wp.Scalar]` generics | **still broken** — `WarpCodegenError` at kernel parse (§12.3) |
| radix key dtypes | unchanged — int32/int64/uint32/uint64/float32/float64 only, 4- or 8-byte values. `segmented_sort_pairs` was NOT extended (int32/float32 keys only) |
| no sparse triangular solve | unchanged |
| generic `Any`-typed `@wp.func` wrappers around tile intrinsics | **a wrapper over a *tile* argument** still fails (NVRTC "more than one instance of overloaded function"); the working route is the builtin-capture factory of §2.7. **A wrapper over a per-lane *value* works on Warp 1.17**: `reduce.block_sum` / `block_min` / `block_max` compile at `int32`/`int64`/`uint64`/`float32`/`float64`, `vec2i`, `vec3d`, `mat33`, `spatial_matrix` and a 25-wide vector in one module, on both devices |
| `@wp.kernel(grid_stride=False)` | benchmarked as noise (±5 %, sign flips between runs) — not adopted |

**Three things that make an upgrade's verification honest, all of which default to a *false pass*:**

- **`compute-sanitizer` is at `/usr/local/cuda-12.8/bin/compute-sanitizer`, not on `PATH`**, and must
  run with `--target-processes all` (through `uv` it hops two processes). **A run that instrumented
  *nothing* also prints `ERROR SUMMARY: 0 errors`, so the proof is the slowdown** (~10x) — always
  time the same suite both ways to confirm it actually instrumented something.
- **Both `wp.capture_while` sites sit behind `wp.is_conditional_graph_supported()`**, so on a box
  where that returns `False` the whole CUDA-graph path is skipped and the tests pass green having
  tested only the fallback. Confirm it's actually `True` here with a pytest plugin that
  monkeypatches `wp.capture_while` / `wp.capture_launch` and counts calls.
- **Re-running our own repro script validates the repro, not the upstream bug.** That is exactly how
  the `bsr_mm` misattribution survived two version re-probes. When a workaround's justification is an
  upstream bug nobody else has confirmed, suspect the repro.

**Gates for an upgrade**, all four: the full suite (1953 passed / 1 skipped at the 1.16 bump),
`basedpyright` 0 errors, `zensical build --strict` clean, `tests.parity` with an unchanged pair count.
The `pyproject.toml` specifier stays `warp-lang>=1.15` where no newer-only API is used — the pin
lives in `uv.lock`.

**Also re-probe any tuning constant** (§9), and note mkdocstrings renders triwarp's **source-level**
annotations, so Warp's 1.16 change to `repr()` of array annotations never reached the docs.

**Two probe-fixture traps, both of which faked a "DIFFERS" for a whole family:** drawing random input
**inside** a per-device callable hands the two devices different problems (hoist every `rng` draw to
host scope); and `slice_plane` without `merge_vertices()` leaves duplicated boundary vertices, so
harmonic / tutte / arap were being pinned to different boundaries and read as total disagreement
(§7.3).

---

## 13. The cost model (RTX 5090)

### 13.1 Host-side, per call

**Measure `n` calls between two syncs and divide.** A per-call cost taken with a sync *inside* the
loop is up to 14x wrong, because Warp leaves the CUDA mempool release threshold at 0, so every sync
drains the pool and the next allocation is cold. *(Raising the release threshold is 1.00-1.02x on
real wrappers **at small sizes** and **1.5-1.7x on a few-launch call at ~1 M faces and above**
(`cotmatrix`, `crease_edges` at `dragon` / `lucy`, §16.27): a drained pool's cost grows with the
allocation's size, so a call allocating hundreds of MB pays it on every harness round. triwarp keeps
Warp's default -- the setting is process-wide -- and the lever is allocating less, §16.27.)*

| primitive (correct regime) | cost |
|---|---|
| `wp.launch` / `wp.launch_tiled` | **11.8 / 12.2 µs**, independent of `dim` |
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
| `radix_sort_pairs(..., end_bit=b)`, `uint64` keys, graph-replayed | **43.4 against 89.5 µs** at 24 576 keys, 24 bits; 87.8 / 120.7 at 491 520, 34 bits; 310 / 367 at 3 M, 42 bits (§16.23) |
| a cached `wp.map` call (Python overhead above the kernel) | ~11 µs |
| a host readback | ~0.1 ms *queued*, **14.3 µs isolated** — the 0.1 ms is the pipeline drain in front of it, not its own cost (§16.7), so price it by what is queued |
| `wp.synchronize_device` | 1.1 µs |
| a **replayed** kernel in a captured chain | **1.17 µs**, exactly linear from 1 to 12 kernels |

**A Warp-typed value is not a Python number, and its operators are ~370x a Python float's.** This
is a *second* per-call cost model, orthogonal to the table above.
`warp._src.types.scalar_base.__add__` is `return warp.add(self, y)` — Warp's Python-scope **builtin
dispatch**, which runs `inspect.signature().bind()` per operand:

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

Four things this settles, and the first is why the hazard is narrower than it looks:

- **Only `+ - *` and the explicit builtins are silent.** `//` and `%` on a Warp scalar raise, and
  `wp.zeros(wp.int32(n))` raises `'int32' object is not iterable`, so those cannot hide. Comparison
  and construction are effectively free. **Arithmetic and array slicing are the whole of it**, which
  is what makes **check 26** (§4.5) a clean scan with no legitimate instance.
- **A slice is three dispatches, not one**, because `wp.array.__getitem__` forms `stop - start` and
  `int(strides) * start` internally — so a *partially* typed slice still costs 5-8x, and a scan has
  to read `arr[K:]` and `arr[:K]` as well as the symmetric form.
- **`wp.constant(x)` does not make a value Warp-typed and `wp.float32(x)` does not round it.**
  `wp.constant(7)` is a plain `int` (§12.6) and `scalar_base.__init__` is `self.value = x`, so
  `wp.float32(2.0 * math.sqrt(...))` keeps every float64 digit and only rounds when it is marshalled
  into a launch — which `wp.launch` does for a plain float anyway. Wrapping a host intermediate buys
  nothing and costs a dispatch per subsequent operation.
- **Vector arithmetic takes a different path and is a smaller, separate hazard.** `vec_t.__sub__` is
  `_binary_op(self, warp.sub, y, vec_t)`, which for two same-typed vectors runs a Python component
  loop rather than dispatching — and is **invisible to the `Function.__call__` census** in §15.11.

**The cheap replacements, measured.** `float(wp.length(upper - lower))` -> `math.dist(lower, upper)`
is **4.9x**, and a `wp.vec3` indexes to a native Python `float`, which is what makes `math.dist` and
the plain `min` / `max` builtins applicable to one at all. `wp.min(a, b), wp.max(c, d)` ->
componentwise `wp.vec3(min(...), ...)` is **7.6x** and byte-identical. **Quote the end-to-end number
beside the expression's**, because a public wrapper adds its own floor: `bounds.aabb_union` measures
**3.55x** rather than 7.6x, the difference being `require_same_device` and the Python call itself,
and `bounds.enclosing_diagonal` at 200 000 points **1.29x**; `points.plane_basis`, untouched, was
carried through the same runs as a control and stayed flat. Where the NumPy buffer is already in
hand the win is larger — `bounds.aabb`'s corners are a readback, so `enclosing_diagonal`'s two
`wp.vec3` constructions plus subtraction plus `wp.length` become one `np.linalg.norm`, **13.4x**.
**`math.dist` computes in float64 where `wp.length` is float32**, a ~2e-08 relative shift and the
correctly-rounded answer for float32 corners; that is inside every tolerance in the suite but it is
not byte-identical, so gate such a change by running both devices rather than by byte-comparison.

**And this partly contradicts §3.8.** That section lists "NumPy standing in for a Warp Python-scope
equivalent that exists" as a defect. For `length`, `min` / `max`, `inverse` and vector arithmetic the
measurement runs the other way by 3-37x, because those are *host* operations either way and Warp's
route to them is a builtin dispatch. §3.8's rule stands for what it was written about — not forcing
a NumPy round trip or naming `np.ndarray` in a public signature — and `wp.cross` remains the genuine
case where Warp wins. **Decide per operation, from the table, not from the rule's direction.**

The launch and allocation rows were re-measured on Warp 1.17 and each is 20-25 % above what this
table carried from an earlier version; the cost model is still accurate because every term moved
together. **The three rows in bold are the ones that change decisions**, and all three were being
reached for as though they were free: `wp.utils.array_sum` is 3.4x a launch, which is why four of
them were 62 % of `measures.moments` (§16.4); `wp.utils.array_cast` is 1.8x a launch for a copy a
plain kernel does identically, which is why `flatnonzero` stopped using it; and an *isolated*
readback is 14.3 µs rather than 0.1 ms, so a function that already synchronized is not paying 0.1 ms
for its second read.

A host-cost model of `allocations × per-call cost + kernels × 9.7 µs` accounts for 84-122% of a
wrapper's measured host time, verified on seven wrappers spanning 0.3-2.9 ms.

**Launch marshalling is ~1.0 µs per argument, linear, identical on both devices** — measured from 2
to 28 arguments, 14.8 µs at the low end and 41.1 at the high (CUDA mins; CPU within a microsecond of
it throughout). So the often-quoted "~32 µs per `wp.launch`" is the *mean* kernel's launch (mean
argument count 5.1), not a constant. A `@wp.struct` bundle collapses it: a 25-argument kernel
against the identical kernel taking one bundle measures roughly 0.4x median at every size, a flat
~25 µs saving regardless of `dim`. Building the bundle costs ~2.6 µs. Rule and eligibility: §2.8.
Only 18 of 440 triwarp kernels take ≥12 arguments.

**A generic kernel costs a further ~12 µs of host-side overload resolution on every launch — roughly
double the host cost of a concrete one — and `wp.Float` / `wp.Scalar` cost exactly what `Any`
costs.** `wp.launch` runs `infer_argument_types` over the *whole* argument list before looking the
overload up, so the cost scales with how many parameters are generic, not with which spelling names
them. This is flat in `dim`, so it is a genuine per-launch cost, not a first-call effect. **The fix
is one line per module: `wp.overload()` already *returns* the concrete `wp.Kernel`, and the
registration code was calling it and discarding the result.** Landed across all 42 generic launch
sites; verified end-to-end against a detached baseline worktree across ~20 representative wrappers,
19 of 20 improved (typically 1.1-1.6x) and none regressed — mechanism and rule at §2.5 and §16.0.
**A missing dtype now raises rather than silently rebuilding the module.**

**A cached `wp.map` call carries the same kind of overhead**, roughly 1.8x the launch it wraps
(~11 µs) — that is what §3.5's `return_kernel=True` hoist removes.

**Per-segment packing costs are host constants and flat in the data** — the same rows measure
1.46 ms at 0.07 MB total and 2.42 ms at 268 MB, a 4 000x range:

| operation | per segment |
|---|---|
| `wp.copy` (`pack_1d_arrays`, `concatenate`) | **6.02 µs** |
| `wp.clone` (`split(copy=True)`) | **15.06 µs** — a ~10 µs allocation plus a ~6 µs copy |
| a `wp.array` slice view (`split(copy=False)`) | **3.63 µs** |
| one whole-buffer `wp.copy`, 48 903 to 2 614 242 elements | **0.010-0.015 ms**, flat |

So the vast majority of a many-segment pack is per-call overhead, not the data volume.

**A `wp.array` view can be stamped instead of sliced (round 23), and `array.split` does.** A
slice's state is 20 plain attributes in its `__dict__`, four of them per view (`ptr`, `shape`,
`size`, `capacity`). `object.__new__(wp.array)` plus a copy of one *fresh* template slice's
`__dict__` with those four patched is the same object -- `_ref` keeps the base alive -- at ~0.4 µs a
segment against a slice's 3-3.5 (7-9x). The template must be made inside the call: a reused view
may carry a cached `ctype` holding its own pointer. A trailing empty segment (`arr[n:n]`) stamps
safely. `requires_grad`, non-contiguous and subclass bases fall back to real slicing.
`test_segment_views_match_real_slices` pins the layout key by key, so a Warp release that renames
an attribute fails there. `split(copy=True)` keeps one allocation per segment (its contract) and
fills them all with one `unpack_segment_words` launch from 32 segments: 2.5x. `combine.split
[parts_1024]` 8.7-9.5 -> 2.1-2.2 ms. **`_device.require_same_device` scans unlabelled first** and
builds labels only once a mismatch is known: it had built one f-string per list element, about
half of `pack_1d_arrays` at 2 048 segments (525 -> 44 µs over a 2 048-array list).

**A "Warp has no X" comment is a claim to probe, not a fact to inherit** — and this is the worked
case. `_pack_segments` carried *"there is no segmented alternative: Warp has no array-of-arrays and
a kernel cannot dereference a raw pointer"* through several optimization rounds as settled. Warp
has an array-of-arrays: a `@wp.struct` may carry a `wp.array` field, and a `wp.array` of that
struct is a descriptor table a kernel indexes as `segments[s].data[k]`. The probe that refuted it is
fifteen lines (§10 says the same about introspecting a builtin before planning around its absence).

One launch over that table replaces the whole copy loop, and is flat in the segment count *and* in
the total size where the loop is linear in the segments — so the whole choice is a threshold,
`array.PACK_SEGMENTS_KERNEL_FROM = 32`: a loss below 8, **1.89x at 32, 14.9x at 256, 150x at
4 096**. End to end it is ~5x at 256 segments; the gap is the Python-side per-segment validation
loop and `require_same_device`, which is §3.9's contract rather than a defect.

Two details decide whether it pays, and the first is most of it:

- **Build the descriptor vectorized.** Constructing the struct instances one at a time is most of
  the cost and would give the whole win back; one NumPy structured array through the struct's own
  `numpy_dtype()`, uploaded once, is an order of magnitude cheaper.
- **One kernel serves every dtype, not a table of them.** Declare the descriptor's array field
  `wp.array[wp.int32]`, point it at the segment's storage with a length in 4-byte *words*, and alias
  the destination the same way (`wp.array(ptr=..., dtype=wp.int32, shape=...)`, because
  `wp.array.view` refuses a dtype of a different size). It is then a byte copy that never names the
  caller's dtype — verified byte-identical for `int32`, `float32`, `float64`, `vec3` and `uint64`.
  A dtype whose itemsize is not a multiple of 4 keeps the loop.

**`split(copy=True)` is not reachable this way and its floor is a *contract*, not a cost.** Its
outputs are `n_segments` separate arrays, so the allocations *are* the return value. The segmented
kernel could fill one buffer and hand back views, but then holding one segment pins the whole
buffer — **the opposite of what `copy=True` promises**, and `copy=False` already exists for a caller
who wants views. Re-proposed and declined again; §15.5's lesson, the refutation was written at the
site the item proposed changing. Other spellings probed and rejected as real losses — a raw-pointer
view that re-implements `wp.array.__getitem__`'s contract, a shared output buffer, dropping
`src_offset=` / `count=` — are written at their sites in `triwarp/array.py`. **The NumPy crossover is
a segment SIZE (~98 kB) and does not move with the segment count.**

**Before doing arithmetic on a `wp.constant` at Python scope, wrap it in `int()` or derive a plain
value once at import.** Indexing a `wp.array` with Warp-typed slot constants — the spelling every
`wp.capture_while` driver used — costs an order of magnitude more than a plain slice, of which the
bound arithmetic alone is most; `kernels/array.LOOP_CONDITION_VIEW` is a plain `slice` **derived**
from the constant so the two cannot drift. §12.6 records that a typed constant is unusable in host
arithmetic (`//` raises); this is the other half, where it works and is slow.

Two more rows worth not re-deriving: `state.assign([0, 1, 0])` is a third of the cost of the three
separate `wp.zeros(1)` / `wp.ones(1)` buffers it replaces, so packing a loop's state into one word
is a saving as well as a convention; and a `wp.array` *view* costs ~3 µs to construct, which is why
a seed launch should take one row view rather than a flattened prefix of it.

**A device reduction costs ~0.10-0.32 ms flat on CUDA regardless of `n`** (launch + 4-byte read),
while a readback scales with bytes copied — the crossover is wherever the copy exceeds ~0.15 ms
(around 200k `int32`/`float32`, 1M `bool`, 16k `vec3d` elements). Below it a reduction launch is
pure overhead; above it the readback grows without bound. **Under ~100k elements both reduction
forms sit at the ~18 µs launch floor**, so small inputs show no difference at all.

### 13.2 Device-side and memory access

| primitive, in-kernel, amortized over 100k iterations | cost |
|---|---|
| dependent L2 load (160 KB / 640 KB working set) | **118 / 147 ns** |
| dependent load, 64 MB working set | 678 ns |
| 15 *independent* loads from one thread | **524 ns total** (~35 ns each) |
| `wp.tile_sum(wp.tile(x))[0]`, `block_dim` 32 / 64 / 128 | **126 / 325 / 369 ns** |
| `wp.tile_scan_exclusive(wp.tile(x))` + `untile`, any width to 32 | **353 ns** |

**These five numbers decide whether a cooperative (one-block, barrier-synchronized) rewrite of a
serial kernel can possibly win, before writing it.** A correct level-synchronous round needs ≥2
barriers plus a prefix scan ≈ 600 ns; if the level's own work is under that, the rewrite loses.
That is exactly how the BFS drain went (§14.9).

Two facts about the primitives themselves:

- **`wp.tile_sum(wp.tile(x))[0]` is a genuine block-wide broadcast reduction** — correct on *every*
  lane, not just lane 0, and it syncs before and after, so it also doubles as a usable full block
  barrier (as does `tile_scan_exclusive`).
- **Tile ops are legal inside a dynamic `while` loop** on CUDA and behave correctly across lanes.
  Keep the loop condition block-uniform (read it from a global every lane reads after a barrier, or
  from registers every lane updates identically) — `__syncthreads` in divergent flow is UB.

**A single thread's throughput is ~35 ns per independent memory op**, so a serial pointer-chasing
kernel is usually throughput bound rather than latency bound; check that before trying to hide a
stall.

**A `wp.hash_grid_query` cell probe costs ~600 linear-scan point tests** (a hash plus two dependent,
uncoalesced global loads per probe, against a broadcast-out-of-L2 linear scan). Consequence: once the
search radius outgrows a couple of cell widths, an exact O(n) scan is genuinely cheaper than widening
the walk — the break-even span scales as `n^(1/3)`, which is why `_knn_widest_grid_radius` is
`n`-aware rather than a fixed constant.

**`wp.launch_tiled` with every lane walking the whole chunk and lane 0 doing the atomics is FASTER
than one thread per chunk**, even though it looks 64x redundant — all lanes read the same address at
each step so the loads broadcast, where one-thread-per-chunk gives each lane its own run and the
reads stop coalescing. The redundant arithmetic is free; these reductions are memory-bound. See
`accumulate_procrustes_moments`.

**Dropping a `sqrt` from a ball query's narrow phase is flat** — the root costs nothing against the
candidate walk's memory traffic. Declined on that count *and* because `wp.length` vs `wp.length_sq`
are not the same predicate in float32 (§12.4).

**SHIPPED — a single-slot atomic reduction serializes on one address, so its cost is linear in the
launch; converting it to a lane-partition + `wp.tile_sum` + single guarded commit is a large,
real win that grows with the launch size** (roughly 1x at a few thousand elements up to 10-30x at a
million). The converted form needs no CPU/CUDA branch, because its lanes partition a chunk the block
already owns with a `wp.block_dim()` stride (§2.2). **Do not convert a *conditional* atomic** (a
compaction cursor, a change flag, a rare-event counter): contention there is proportional to hits,
not to the launch, so there is nothing to win.

**The same finding recurs at *one atomic per block*, and it names the real variable — the block
count, not the redundant lane arithmetic.** Several kernels already had every lane walk a whole
chunk and only lane 0 publish, so they looked done — but they still issued one atomic per *block*,
and the block count itself was the remaining cost. Re-launching at a finer block grain (the same
fold width the reduce module uses) cuts the block count substantially and is worth a further
several-x at large sizes. **Isolating the 64-fold "wasted" lane arithmetic alone recovers almost
nothing** — so the shape to look for is not "lanes doing discarded work" but "how many blocks reach
the accumulator", and the lever is the fold width, not the redundancy.

**The shape is greppable**: look for `wp.launch_tiled` at a `dim` sized to the *item* count (rather
than the reduce module's own block-count helper) with a `lane == 0` commit. Not every kernel matching
that `dim` shape matches the defect, though — a kernel that partitions the *outer* work at a constant
stride is a different, `_sliced`-paired case, where changing the fold is a rewrite of the partition.
**That rewrite was tried for one such case and came back flat, which sets a threshold for the whole
family: the quantity the fold reduces is `blocks x accumulator slots`, and it needs to be around 1e5
or more before there is anything to win** — compute that product before proposing this class of
rewrite.

Two implementation notes for the next conversion of this shape: a wide accumulator needs the fold
*more*, not less (more per-block reductions to amortize); and because `wp.tile_sum` is block
collective, all of them run *outside* the `lane == 0` guard and only the final commit is inside it —
assemble the whole vector/matrix from the tile sums first, since `wp.atomic_add` reads trailing
indices as array dimensions, not vector components.

**Flattening a reduction to its launch floor can make a *neighbouring* fusion worth doing, and a
written decline can expire silently as a result.** A fusion between a reduction and an adjacent
kernel was declined when the reduction was most of the pair's cost — once the reduction became a
flat launch floor, the same fusion became a large win, output bit-identical. **After landing a large
win on a kernel, re-read the declines about the launches either side of it** (§9).

**A single-address `float64` `wp.atomic_add` serializes the launch entirely** — a global dot needs a
two-stage reduction, not a single accumulator.

**Padding must be a hole, not a value.** Pointing padded rows at a dummy valid index makes
`bsr_from_triplets`-style accumulation collide on one entry and serialize; sending them out of range
(silently dropped) is the fix. Same trap as §12.7's zero-padded triplets, from the other direction —
**look for it whenever padding has a *value* rather than being a hole.**

**Three tiling antipatterns, all measured in `kernels/reduce.py`:**

1. **A `wp.tile_load` kernel below one tile is a large loss** (~49x on one case). When the *reduced
   extent* is under the tile width, `launch_tiled` still gives every block many lanes, and all of
   them redundantly walk the same short row — fixed by falling back to one plain thread per output
   row when the reduced extent is small. **But the opposite reduction axis must stay tiled** — the
   same fallback on the other axis is a large loss the other way, because there are too few outputs
   to keep the device busy serially. **The dispatch key is the reduced extent, not the axis**, and
   this recurs one rank up for 2-D tables with a short trailing dimension — fixed the same way,
   gated on the trailing extent (and contiguity, since flattening a non-contiguous view raises).
2. **One atomic per tile does not scale** (~4.9x lost at large sizes). Global 1-D reductions issuing
   one atomic per 64-element block put far too many blocks on one accumulator address at scale;
   folding several tiles into a register before the atomic recovers most of it. Swept fold widths
   found a middle value that never loses across the whole size range tested.
3. **`tile_chunk` reports what is left to the end of the array, not the block's share of it, and
   clamping is the caller's job.** Its own docstring says so; the tile-load factories satisfy it
   *implicitly*, through a fixed `TILES_PER_BLOCK_1D` loop that cannot overrun. A kernel whose loop
   is bounded by `remaining` instead — any `for k in range(t, remaining, wp.block_dim())` — must
   write `remaining = wp.min(remaining, ITEMS_PER_BLOCK_1D)` or block 0 walks the whole array. It
   fails loudly for a sum and **silently for min/max/any/all**, where re-reading elements another
   block already read is idempotent. Look for it whenever a new reduction kernel does not use
   `wp.tile_load`.

   **The coupling hazard this creates: changing how much work a kernel does per block silently
   breaks any *other* module that launches it directly and computes its own `dim`.** Fixed by
   exporting the block-count helper for callers to reuse. **`grep` for direct launches of a kernel
   before changing its per-block contract.**

**A fused fold's tuning variable is the fold width, and a kernel that *also* writes per element
must keep its per-element dimension in the grid.** Folding a counting reduction into a kernel that
already computes the same predicate is free arithmetic — but launching it at the reduce module's
`blocks_1d` grain is not, because `ITEMS_PER_BLOCK_1D` is 1 024, so each block owns 1 024 elements
and the grid collapses. Measured on `homology.dual_candidate_mask`, which writes one `wp.bool` per
edge *and* folds the interior-edge count: at 140 964 edges the wide fold is **137 blocks — under
one per SM on a 170-SM device** — against 2 188 blocks at one tile per block. The narrow form
launches `dim = ceil(n / TILE_1D)` with `tile_chunk(n, chunk, TILE_1D)`, which keeps one element
per lane and still commits one atomic per block. Whole-call effect of getting it wrong: 0.92-0.95x,
recovered to parity by the one-token change of the fold width.

This is §2.3's occupancy rule reaching a kernel that is *not* a block-per-item candidate: the test
is not "does this kernel have an outer dimension" but "does anything other than the reduction need
one thread per element". A pure reduction takes the wide fold; a reduction fused onto a map does
not.

**Two more shape facts:** a plain `wp.tile(vec3)` decomposes to a scalar tile, but
`wp.tile(v, preserve_type=True)` keeps the vector, and `wp.tile_sum` then reduces it componentwise
in **one** tree -- bit-identical to one reduction per component, on both devices, for vectors,
matrices and `wp.types.vector(length=N)`. So a block that folds several same-dtype quantities packs
them into one vector and pays one barrier (`reduce.block_sum`); the per-component spelling the tree
used to carry cost 25 barriers in `accumulate_procrustes_moments` and 44 in
`accumulate_point_to_plane`, now one and three. And `wp.array.view(wp.float32)` on a vec3 array
gives a zero-copy `(n, 3)` view for `reduce.minmax`.

### 13.3 Tuning constants are per-device

`ITEMS_PER_SLICE` (elements per thread in the lane-free strided-slice reductions) has a
device-dependent optimum: **CUDA wants long slices, CPU wants short ones** — shipped as
`ITEMS_PER_SLICE_CUDA = 128` / `ITEMS_PER_SLICE_CPU = 32` behind `_device.items_per_slice(device)`.
**Slice length is not one number even within a device**: 32 suits reductions into one or a few
accumulators, while a *per-query* reduction wants ~128 (`proximity.ITEMS_PER_QUERY_SLICE`) because
the query dimension already fills the device and a short slice only multiplies atomics.

**When adding a device-dispatched path, re-ask which device still reaches each branch, and re-sweep
the constants that branch reads** — the old measurement may no longer describe any live call site.
`ITEMS_PER_SLICE` had been tuned against two call sites that, after a later change, stopped reaching
the sliced form on CUDA at all.

**Sweep the values you didn't try the first time when re-probing after an upgrade, or the re-probe
inherits the original's blind spot.** Two constants re-probed after a Warp upgrade gave opposite
outcomes: one reproduced exactly (no change needed); the other's original sweep had only tried two
candidate values and read as a clean two-bracket split, but a finer sweep revealed a *third* bracket
in between, which the coarse sweep's two points straddled — costing a real loss at exactly the size
the old crossover was tuned around. **A constant whose sweep sampled only two values has not been
shown to be a two-bracket problem.**
---

## 14. Kernel-shape verdicts

### 14.1 Block-per-item (one block per item, lanes stride the inner sequence)

**It wins where the outer dimension alone starves the device and loses where it does not** — the
criterion and the win/loss table are §2.3. Three shapes in the tree take it:
`kernels/visibility.py::obscurance` (block per point, lanes over its ray bundle, 3.2-11.8x,
`block_dim=64`), `shape_diameter` alongside it, and `points.farthest_point_sample` as **one
persistent block** — `launch_tiled(dim=(1,))`, lanes striding the cloud, `wp.tile_max` over a packed
key serving as both the argmax *and* the barrier, with an identical tie-break.

**FPS wins where the hole DP lost**, and the contrast is the rule: its per-iteration work is `n`
distances, so one SM is enough and what it removes is two replayed kernels of launch latency per
dependent round. `block_dim` 1024 on large clouds, 256 on small; 64 for the visibility bundle.

**Four kernels keep the arg-strided form deliberately**, each annotated with its number: they
already carry a *slice* dimension, so the outer dimension is not what starves the device
(`points.hull_support_extremes` is 2.3x at 5 000 points and a **2-8x loss** at 200 000).

### 14.2 Cooperative BVH walks

**`tile_bvh_query_aabb` beats the hash grid in the narrow-query regime and regresses above the
crossover** — roughly 4-8 k concurrent queries for a ball of the cell width, 16-65 k for twice it.
Measured on one cloud/radius/query set with all walkers agreeing exactly:

| walker | ball of `r` | ball of `2r` |
|---|---|---|
| `wp.HashGrid` serial (incumbent) | 70-83 µs | 359-394 µs |
| `wp.Bvh` **serial** | 132-148 µs (**worse**) | 317-399 µs |
| `wp.Bvh` **tiled** | **29.5-67.6 µs** | **40-99 µs** |
| hashed cell grid, cooperative | 8.6-12.2 µs | 18.3-28.1 µs |

The serial BVH being *worse* than the hash grid is the tell: **the win is the tiled traversal, not
the structure** — and it spends 32 lanes on a query the device could already saturate, which is what
the crossover is.

**Do not re-run the tree-wide sweep**: exactly one of three hash-grid usages could take it.
`neighbors.query_*` runs past the crossover, `poisson_fem.refinement_oracle` has no block to
cooperate over (`warp.fem` owns the launch shape), and `ball_pivoting`'s pivot search qualified on
its few-hundred-edge front. Its empty-ball test stayed a per-lane serial *hash-grid* query — at that
point each lane tests a different ball, so there is nothing to cooperate on.

**The cell-list numbers bound what a bespoke index could be worth**: a hand-rolled cell list is
1.3-1.7x faster even single-threaded (Warp's iterator pays for generality this workload does not
use), and 7-30x with lanes splitting the cells. A *dense* grid cannot ship — for a surface cloud the
cell count grows as `n^1.5` — so store the packed cell key per entry and compare it exactly. **Only
build one if a measurement shows the query is *still* the bottleneck**; its extra margin is probably
unspendable against a kernel's own throughput ceiling.

**A thread-per-query BVH launch is usually load-imbalanced, not under-pruned. Histogram the
per-thread candidate count before optimizing one.** On a scan mesh against a translated copy, 0.16 %
of candidates survive the box prune (the leaf test is not the cost), 98.2 % of faces return no
candidate at all, and 0.5 % of faces carry half the traversal. The fix is a capped thread pass that
appends stragglers to a work list, then one warp per straggler. Two levers declined with it:
`block_dim` (256, the default, wins at every value from 32) and tightening the query margin — the
vertex bound *equals* the answer to all 16 digits, so the query only confirms what the bound found.
The imbalance does not necessarily persist at scale (§16.6).

**A BVH walk's device time is sensitive to how its loop is spelled, by up to 5x, with no resource
difference to show for it** (§16.23). Five count-only walks over one mesh's BVH, identical
candidates, 37-38 registers, the same 33 KB shared traversal stack and no spills: `mesh_query_aabb`
with a stored box and no cap 97 us, the same with a literal `c < 32` cap 453 us, `bvh_query_aabb` 522
us whether its box was stored or formed in the kernel, and `mesh_aabb_collect` (a runtime cap) 98 us
with a stored box but 455 us with the box formed from the face corners. Not a rule to spell around --
it is code-generation variance -- but it means **a new or fused walk kernel is timed against the one
it replaces, on the device**. The lead it opened for `mesh_to_mesh_distance` is **closed**
(round 22): its capped walk with `mesh_query_aabb` in place of `bvh_query_aabb`, and with the grown
boxes precomputed, measured 0.93-1.08x with identical overflow counts -- the variance is not present
in that kernel's loop.

### 14.3 CUDA graph capture

**Capture and argument bundling address different loops, and neither substitutes for the other:
capture pays on a launch sequence that repeats identically, a bundle on one that runs once.**
Recording costs at least what issuing the launches costs, because capture intercepts each one:

| arm | vs loose arguments |
|---|---|
| `@wp.struct` bundle | **1.88x** |
| capture-and-replay-**once** | **0.84x — a loss** |
| replay of an already-recorded sequence | **4.29x** |

That last row is where capture's reputation comes from and it is unreachable without a *repeated*
sequence. **Do not reach for capture on a once-through Python loop** — recording the whole of it
and replaying once is the 0.84x row, and the packing family reproduces it (a pack's segment
pointers change every call, so the recording is never reused; an earlier claim of a crossover at
~1024 segments **does not reproduce**).

**But a sequence that is not repeated can often be *made* repeated, and that is the lever this
table hides.** A long chain of launches that differ only in a loop counter is not an identical
sequence — until the counter moves onto the device. Then a *group* of the chain is recorded once
and replayed to cover the whole of it, with a `dim=1` kernel stepping the counter as the graph's
last node (replays serialize on the stream, so the group's kernels have all read the old value
before it moves). Measured on the hole-fill span sweep, whose launches differed in one `wp.int32`:
**510 sequential launches cost 6.45 ms issued and 1.10 ms as an 8-launch group replayed 64 times,
5.9x**, recording included. Three things make it work, and the first two are what to check next
time:

- **The grid must be fixed across replays**, so the recorded group is sized for the *first* group
  and over-covers every later one. That is free here — a span kernel's cost is flat in its unused
  grid width (0.91-1.02x for the same sweep issued at maximal width, device-bound and host-bound
  alike) — but it is only free where the surplus threads exit on a guard they would reach anyway.
- **The group size is a shallow optimum**, because recording costs one ordinary launch per span in
  it while a small group pays one extra counter kernel per replay: 8 won at every rim length
  probed, 64 and up lose.
- **It has an upper bound, and it is not the chain length.** Once the kernels cover their own
  launches the host was never the critical path and replay only adds the counter kernel — a
  2-4 % loss. The gate is the *device work each launch carries* (§16.6), not how many there are.

**`wp.capture_begin` is available on CUDA only**, so a captured path always needs the plain loop as
its CPU sibling — which doubles as the byte-identity reference a schedule change has to pass.

**Where capture pays, it pays large**, and what made `quadric_decimate`'s pass legal is that the
pass is **flat in the mesh size**, so replaying every pass at the pass-0 width costs ~1.01x. A
captured wrapper chain also pays for its Python once, so the wrapper chains stayed and the capture
removed their cost. §12.6 has what blocks a capture; §15.10 has why a captured function cannot be
attributed with `wp.timing_begin`.

**A `wp.capture_while` body may not allocate, and `wp.utils.array_scan` does** (CUB scratch, per
call): recording one raises `Conditional body graph contains an unsupported operation (memory
allocation)`, where the same scan in a *plain* `ScopedCapture` is legal. Probed in isolation:
`fill_`, `zero_` and `radix_sort_pairs` record into a conditional body, `array_scan` does not. So a
loop whose round contains a scan -- the flip rounds' regroup, `quadric_decimate`'s pass -- is
recorded as a plain graph and replayed from the host, keeping its one 4-byte termination read per
round; that is still the manufactured-repetition win above (§16.14: 1.48x on a 50-round flip call).

**`wp.capture_while` is slower than a batched host loop** where the per-iteration conditional-graph
overhead exceeds the sync it removes (ball pivoting batches 8 waves per readback). Nesting one
inside a capture is fine.

**The unexplored lever is the reverse:** a *repeated* wrapper loop issuing an identical sequence
that is not yet captured, and `wp.capture_if` (a device-side conditional, unused here) for a stage
that currently spends a host readback deciding. **And the manufactured-repetition trick above is
worth a sweep of its own** — grep for a Python `for`/`while` whose body is one launch and whose
only per-iteration argument is the loop variable.

### 14.4 Tile solves: the crossover is K ≥ 16-32

Batched K×K SPD solves, one thread per system (Cholesky in registers on a `wp.matrix`) against one
block per system (`wp.tile_cholesky` + `wp.tile_cholesky_solve`), values cross-checked. Subtract the
harness floor (~14 µs for an empty `dim=1` launch plus `synchronize`) or small-K rows read as false
parity:

| K | N | per-thread | tiled | |
|---|---|---|---|---|
| 6 | 65536 | 8.5 | 38.5 | tiles **4.5x slower** |
| 8 | 65536 | 17.3 | 53.6 | tiles 3.1x slower |
| 16 | 1024 | 27.7 | 7.2 | tiles 3.9x faster |
| 16 | 65536 | 54.1 | 100.6 | tiles 1.9x slower |
| 32 | 1024 | 585.6 | 15.6 | tiles 38x faster |
| 64 | 1024 | 3989 | 85 | tiles 47x faster |

Tiles win only when **occupancy-starved**: few systems, large matrix. **triwarp's only dense solves
are K=6 (N=1) and K=5 (N=n_vertices), both on the wrong side**, so "rewrite the small dense solves
as tiles for a CUDA-only fast path" is refuted — do not re-propose it. It is *not* blocked by the
CPU lane constraint; there was no opportunity to lose.

### 14.5 NumPy readback vs device reduction: 3 of 7 converted

Interleaved A/B on the scan meshes, **on both devices**. A = readback + numpy, B = device reduction.
**The CPU axis is what rejected 4 of 7**, so the rule is §9's: decide on CUDA, but measure both.

**CONVERTED** (win on both devices): reusing an existing perimeter kernel instead of a per-loop
`.numpy()`; `.numpy().sum(axis=0)` → `wp.utils.array_sum`, which reduces a `wp.vec3d` array
componentwise and so needs no kernel; and one weighted-branch iso-value reduction.

**REJECTED — the current numpy code is faster:** edge-range validation (a 14.7x CUDA win against a
**20x CPU loss**); `bool(mask.numpy().any())` (A wins everywhere — a bool array is 1 byte per
element, so the copy never clears the launch cost); a `.sum()` + `.min()` pair that also re-uploads;
and a uniform iso-value branch.

**Two method notes that cost a wasted first run**: sequential A-then-B timing produced non-monotonic
garbage with a recurring artifact, and pre-allocating B's scratch outside the timed callable
flatters it. **And grep the whole package for a private helper's callers, not just its own module** —
a cross-module caller failed only in the full suite, and basedpyright cannot see it either (§8).

**The axis is often not the one it looks like.** The per-loop readback's axis was the *loop count*,
not the vertex buffer: on CPU `vertices.numpy()` is a zero-copy view, so the "whole vertex buffer
copy" never existed and what cost hundreds of milliseconds was iterating thousands of loops in
Python with one `.numpy()` each.

### 14.6 Readbacks inside device loops

A per-pass readback costs ~0.1 ms while one extra loop pass costs 0.9-2.4 ms, so **any scheme that
trades redundant passes for fewer syncs loses**. Checking every 2 passes instead of every pass is
+5 % to +30 % with bit-identical output; `cg(check_every=25/50)` is +0 % to +6 %, because the
overshoot adds real CG iterations.

`cg(check_every=0)` is the exception on paper — it converges device-side with zero host readbacks —
**but it is not a lever in the harness**: an isolated probe read 2.1-2.3x where the harness reads
1.05-1.23x and **0.94x** on the conditioning rows (§15.9). Do not re-try it. It also changes `cg`'s
return from scalars to device arrays and depends on conditional-graph support.

**Batching convergence checks loses whenever an extra iteration runs a real kernel** (a BVH query, a
whole-graph hook): keep per-iteration 4-8-byte readbacks there and fuse multiple flags into one
buffer instead.

**Before removing a host sync from a loop, price one loop iteration first.** Only remove it if the
host could actually run ahead; if the next iteration depends on this one's result, it cannot. A
readback's *self-time* in a profile is the queue depth in front of it, not its own cost, so profiles
make syncs look far more expensive than removing them turns out to be.

### 14.7 Per-device algorithm choice (rare, and the criterion is asymptotic work)

Two functions branch on the *device* rather than on a constant, both because the parallel form does
**asymptotically more work** to expose parallelism — so on a backend that runs a launch grid as one
serial loop there is no GPU win being paid for.

- **`_device.prefers_tiled_reduction`** — a *correctness* branch (§12.2), not a performance one.
- **`polyline_downsample`'s pointer-doubled greedy walk** (`_DOWNSAMPLE_DOUBLING_FROM = 2048`, CUDA
  only): the kept set is the orbit of point 0 under "the next point at least `step` further along",
  so build that step function for every point at once and pointer-double it. **Crossover ~2 048** since
  round 23 (it was 8 192 when a round cost two launches); on CPU it loses at *every* size (30x at 65 536) because it does
  `n log n` work where the serial form does `n` — and Warp's CPU walk is *faster than CUDA's*, no
  launch to issue and a cache-friendly stride.
    - **Exactness is the claim**: the successor search evaluates the same float32 predicate the walk
      does, so the masks agree bit for bit; `cum[mid] - step` would *not* be the same predicate. A
      large-`n` NumPy oracle is unavailable (a float64 sequential `cumsum` against Warp's float32
      tree scan differs by the same order as the gaps between decisions), so exactness is pinned
      triwarp-against-triwarp and the large-`n` test is invariants only.
    - **Re-probed in round 23: the crossover is ~2 048**, and `_DOWNSAMPLE_DOUBLING_FROM` is now
      2048. `double_greedy_orbit` fuses a round into one launch and chases up to
      `POINTER_JUMP_MAX_HOPS = 16` pointers per launch (`rim_long` 20 -> 8 launches); serial over
      jumped reads 0.72x at 512 points, 0.89x at 1 024, 1.15x at 2 048, 2.59x at 8 192, 15.4x at
      65 536.

### 14.8 Solvers: the cycle is launch-bound

**Any smoother that buys iterations with launches loses on this machine** -- inside a V-cycle. A
single-level polynomial *preconditioner* is the opposite trade and wins (§16.15): a CG iteration is
half a dozen launches and two reductions, a polynomial step one fused mat-vec. A Chebyshev multigrid
smoother was built and reverted: interleaved against damped Jacobi over five systems, Jacobi wins
four of five, the one loss is 1.03x and the worst Chebyshev cell is a 2.07x regression. The same
reasoning predicts the sweep-count table being flat.

**But the launch argument overstates the case by ~4x, and the verdict survives on other ground.**
One V-cycle apply issues 34 launches of which the smoother is 9, so swapping every Jacobi sweep for
a Chebyshev step raises the *cycle's* launch count by 1.13x, not 1.5x, against a ~1.03x iteration
decrease. **What still refutes it is interval robustness**: no single `(degree, interval)` is best
everywhere and `rho/5` is a cliff of up to 17x, so there is no safe default to ship. Stop quoting
"the cycle is launch-bound" as the reason.

Two traps from that work: **a small synthetic system lied** (a 576-unknown grid Laplacian gave a 7x
iteration reduction that vanished on the real meshes); and **`sweeps` is a default argument frozen
at import**, so patching the module constant does nothing and the first probe returned an identical
iteration count at every value while looking like a working measurement.

A direct GPU factorization and the conditioning-flatness evidence are §16.8.

### 14.9 Refuted, with the code written — do not re-propose

**Read §14.11 first.** Two of the entries below are single-block rewrites of a whole problem, and
one reader in three takes them as "cooperative kernels lose here". The distinction that decides it
is *tiling*: putting the whole sequential problem on one block loses, and keeping the grid full
while trading a launch for a block barrier is the biggest win in this package.

- **A single-block cooperative BFS drain.** Built, byte-identical, and a large loss. The serial
  drain is memory-throughput bound on one thread, not latency bound, so neither software pipelining
  nor a register-vector batch of loads helped; and on a narrow, non-growing frontier the per-level
  work is smaller than the barrier cost a correct cooperative round needs, so thousands of levels
  put a multi-millisecond floor on synchronization alone. **A per-level dispatch cost only matters
  if the level body actually runs** — on a ribbon graph the parallel level loop emitted 6 nodes of
  40 962 and the rest was the serial drain, so "fuse the level's kernels" was a no-op fix for a
  stage that never executed. (`graph.bfs` has since been deleted — §16.9 — so this is history; the
  lesson is not.)
- **A persistent one-block-per-loop tiled kernel for the Liepa hole-fill DP.** Built,
  byte-identical, and it loses at every rim size, worse as the rim grows: the DP's total work
  outgrows what one SM can do serially long before launch overhead becomes the bottleneck. **Same
  conclusion as the BFS drain from the opposite direction** — one design had too little work per
  level for a block, this one far too much. The only design that would beat both is whole-GPU work
  between cheap level barriers, i.e. a grid-wide barrier, which Warp does not expose (§12.2).
    - **And the *blocked* interval DP over the same recurrence is refuted too, by arithmetic
      rather than by building it** (§16.6). Its schedule is sound — tiles on one tile-diagonal are
      mutually independent — but a tile-diagonal holds `B/tile` tiles where a plain span level
      already launches `B - span` blocks, and that width is what fills the device. **The general
      form: merging `C` dependency levels into one kernel caps the block count at ~`B/C` and
      multiplies the work by ~`C/2`, so it pays only where the unmerged levels were themselves
      under-occupied.** That is the test to apply before citing §14.11 for a new DP. What *did*
      remove the fill sweep's launch cost left the level structure alone and made the launches
      cheaper instead (§14.3).
- **Tile solves at triwarp's own problem size** — §14.4. **A Chebyshev smoother** — §14.8.
- **Voxel aggregation for the multigrid hierarchy**: the geometrically-natural aggregation blows up
  operator complexity far more than the algebraic one already shipped, and a cheaper unsmoothed
  variant does not fix the underlying convergence-rate dependence on mesh resolution.
- **Micro-optimizations of the old Bridson blue-noise propose kernel**, superseded by an algorithm
  change (§16.7) but each a valid null result: cheaper random permutation schemes introduced enough
  bias or overhead to be a net loss; dropping either pruning pass was a large loss, confirming both
  load-bearing; and making an eager shuffle lazy was a wash.
- **A `wp.Stream`-overlap rewrite for any "independent-but-sequential" pair, tree-wide.** Five real
  candidates were built and measured with two streams joined by `wait_stream`: **0.87-1.07x**, never
  a repeatable win. Two structural reasons, not a per-pair fluke. Every branch here is
  host-launch-overhead dominated (§13.1), so there is at most a sliver of device time for a second
  stream to hide behind and the `ScopedStream` bookkeeping costs about as much. And a CG solve
  issues a host readback every `CG_CHECK_EVERY_FALLBACK` iterations, so wrapping a whole solve in
  one stream context still runs every readback to completion before the *next* Python call issues a
  kernel: **sequential Python calls cannot overlap through streams alone** unless the two loops'
  iterations are interleaved at the call site, which is a rewrite of the iteration. **Do not build a
  stream-overlap convention on "these two calls have no data dependency" alone** — check the
  device/wall split first. For a same-operator, multiple-right-hand-side solve, reach for
  `linalg.solve_spd_columns`'s `_BatchedCg` instead; merging their Krylov *subspaces* as well is a
  different mechanism and was removed (§16.8).

### 14.10 Producer-consumer fusion: always fuse; the iteration count only decides whether a benchmark can see it

**Two consecutive launches at the same `dim` where the second reads the first's output only at its
own thread index are fusible, and the fused kernel is faster — every pair measured in this tree, at
every size, 1.04-3.46x.** It removes a launch, an allocation and a full round trip of the
intermediate buffer through global memory. What varies is not whether the fusion wins but whether
any *benchmark* can resolve it, and that is set by the region's share of the call it sits in.

A tree-wide scan finds **93** runs of consecutive same-`dim` launches, of which **89** adjacent
pairs pass the index-locality test. The scan is ~120 lines of `ast` — walk each wrapper's launches
in line order, map each launch's argument expressions onto its kernel's parameter names, and ask
whether every array the first kernel *writes* is read by the second only at a name bound from
`wp.tid()`. Re-derive it rather than re-reading 93 call sites.

**Measure the region, not the call that contains it** (§15.2). A whole-call A/B cannot resolve a
region worth 0.1 % of a CG-dominated solve, and its noise then reads as a verdict: attributing
through the call gave 1.02x and a 25-row harness sweep gave 0.987-1.041x, where the region itself is
1.04-2.65x and never slower. **A cell whose region is a fraction of a percent of the call cannot
report that region's speed — do not let it vote.**

**Two deterministic cross-checks settle such a case faster than any clock** (§15.6): the kernel
count is `base - 1` per fused site, so no solver iteration count moved — the live worry, since a
fusion shifts the answer by 1-3 ulps — and the allocation count is strictly lower by one buffer per
site. Same kernels, same iterations, one fewer allocation: there is no mechanism by which the rest
of the call can get slower, which makes sub-1.0 cells *provably* noise rather than arguably noise.

Three shapes worth knowing before looking for the next one:

- **A fusion can remove one of two intermediate buffers and still be worth 1.79x.**
  `filter_humphrey` applies the operator twice per pass and needs both results, so only one
  disappears.
- **When the middle stage of a three-stage pass blocks the obvious fusion, look across the loop
  boundary.** `filter_normals`' scatter needs the whole seeded buffer, so the fusable pair is the
  normalization with the **next** pass's seed — adjacent only once the first seed is peeled off the
  front of the loop.
- **Two launches of the *same* kernel merge into one wider launch, and that is the biggest win
  available.** A cap kernel run twice per solid, into adjacent blocks of one buffer, becomes one
  `dim=(2, n_cap)` launch with the row index selecting the offset and the winding: 3.4x. **Grep for
  a kernel launched twice in a row with different scalar arguments.**

**Still declined:** two launches in *different branches* are not sequential; and a pair inside a
graph-captured loop is not a candidate, since a replayed launch is ~1.17 µs (§14.3).

**Three traps in the scan itself, because a re-run will hit all three.** Index-locality from an AST
walk is *necessary and not sufficient*: it does not follow `@wp.func` calls, so a kernel reaching a
neighbour's entry through a helper reads as local; it does not see **intervening host work**, so a
pair with a `counts_to_offsets` scan between the launches is reported adjacent and is not fusible;
and a *claim/commit* independent-set pair is never fusible however local it looks, because commit
must see every claim. Verify each candidate by reading both kernel bodies and the wrapper lines
between the launches.

### A fused kernel is not finished until the duplication *inside* it is gone

**Inline any `@wp.func` the fusion leaves with a single caller, then look for what the inlining
exposes.** A helper that is not reused is a name, not an abstraction, and while it stays a helper
the redundancy between the two halves is invisible. In three of eight fused kernels the inlining
exposed a doubled vertex load, a doubled grid index, and a quantity computed twice by two spellings
of the same expression — **that last one took its region from 1.15x to 1.29x, and from a tie to
1.13x at the large end**, so the fusion would otherwise have shipped as "no measurable gain".

**The fix is a local, not a new `@wp.func`** — the redundancy is *within* one kernel, so a helper
would only be a named way to recompute it. Reach for a shared helper when two *kernels* repeat a
run.

**One near-duplicate is deliberate, and removing it was built, measured and reverted.** A scalar-field
CSR row walk spells itself out rather than calling the shared `operator_row`, which promotes the
float32 weight to float64 for a `wp.vec3d` accumulator. Both merges were tried:

- **Keep float32 storage, accumulate in float64** — no extra bytes, and it **does not compile**:
  `wp.float64(w) * f32_value` is a hard parse error. That is §12.4's "scalar arguments must be
  constructed at the input's precision", and it is what makes the two irreducible rather than merely
  inconvenient.
- **Carry the field in float64**, which does let one `Any`-generic helper serve both (seed the
  accumulator with the first term instead of a typed zero — there is no way to spell "zero of
  `Any`'s type" in kernel scope, and `0 + a == a` exactly, so it stays bit-identical). Built, and
  **0.63-0.86x**: the field is one of four streams the walk reads, so doubling its width costs
  bandwidth no launch saving repays, before counting the conversion passes a float64 iterate adds.

**Two kernels that differ only in a dtype are not always mergeable, and when the merge costs memory
traffic the right answer is to duplicate the loop and write down why.** The `Any`-generic form was
also reverted on §4.2 — with the scalar path gone it had exactly one instantiation.

**And check what the fusion killed.** Removing the last launcher of a kernel leaves dead code that
still costs import time (§12.6), and it stales the `wp.map` bookkeeping, since a fused step kernel
is one fewer `wp.map` site (check 23's allowlist named an op that was no longer mapped at all).
**Move the deleted kernel's prose onto what survives** — four comments carried a sign convention, a
normalization precondition and a deliberate non-merge decision, all of which had to be relocated
rather than lost.

---

### 14.11 Blocked wavefront: trade a launch for a block barrier, keeping the grid full

**A dependency chain of `N` sequential steps does not need `N` launches.** Where the chain is over
a *grid* rather than a scalar — a 2-D DP whose cell reads only the cells above and to the left, the
classic shape — tiling it into `T x T` squares and launching one tile-diagonal at a time leaves
only `2 * ceil(N / T)` launches. Every other step becomes a block-local barrier. Measured on
`stitch_loops_min_weight`'s band DP: **2 049 launches to 65, 11.0-13.3x on the DP sweep and up to
6.94x on the whole public call**, byte-identical (§16.12).

**It is the opposite of §14.9's two refuted rewrites, and the difference is the grid.** Those put a
whole sequential problem on **one** block, so the device ran at one SM and the barrier cost bought
nothing; this keeps one block per tile and as many tiles as the diagonal holds, so the concurrency
is what it was and only the *synchronisation* got cheaper. **Ask which one a proposal is before
citing either precedent.**

The conditions, all four of which the band DP meets:

- **The dependency is local and directional** — cell `(i, j)` reads `(i-1, j)` and `(i, j-1)` and
  nothing farther. That is what makes a tile depend only on its up and left neighbours, so a
  tile-diagonal's tiles are mutually independent.
- **The per-step work is small next to a launch.** At ~12 µs a launch against ~126 ns for a
  32-lane `wp.tile_sum` barrier (§13.1, §13.2), the trade is worth ~95x per step before any
  other effect.
- **Warp exposes no barrier, so a block-collective reduction is it.** `wp.tile_sum(wp.tile(x))[0]`
  synchronises *and* orders the global writes the next step reads. **Probe that by removing it**:
  it fails only at 64 lanes and above, because one warp needs no barrier, so a test at the shipped
  32 cannot see a barrier that is gone.
- **The schedule must be provably value-neutral**, which means keeping the cell body in one
  `@wp.func` both kernels call and pinning the fast schedule to the simple one byte-for-byte
  (§2.4). Build the comparison against **one** set of inputs: rebuilding them per arm compares two
  different problems wherever anything upstream uses a float atomic, which read as a dozen
  mismatches here and were the probe's own fixture (§12.10).

**Where to look for the next one:** a Python loop issuing one launch per step whose `dim` is a
*slice* of a 2-D table. **But check the untiled schedule's occupancy first — that is what decides
it, and the fill DP is the counter-example.** Its span sweep is the same *shape*, and tiling it is
refuted (§14.9, §16.6): a span level already launches `B - span` blocks, so a tile-diagonal's
`B/tile` tiles throw away the very parallelism this DP's diagonals never had. The stitch DP won
because an anti-diagonal of O(1)-work cells was under-occupied *before* tiling, so the launch count
was all there was to pay. Where a chain's levels are already wide, the launches get cheaper by being
recorded rather than restructured (§14.3).

## 15. Benchmark and measurement traps

### 15.1 A sudden slowdown is a Warp rebuild until proven otherwise

Before profiling anything, before believing a kernel got slower, before deleting a test for being
slow: a single-digit-second operation that now takes tens of seconds, or a test file whose cost
appears and disappears as you change *which* tests you select, is almost always a module recompile
from an unregistered generic-kernel overload (§2.5) or an undeclared `wp.map` signature (§3.5).
Measured up to 1 200x apart between a fresh test selection and an identical repeat.

Two cheap confirmations:

```bash
# 1. The compile is single-threaded nvcc, so the GPU is idle while the clock runs.
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader   # 0 % during the stall
# 2. Warp says so outright.
uv run python -c "import warp as wp; wp.config.log_level = wp.LOG_DEBUG; ..." 2>&1 \
    | grep -E "Module hash changed, recompiling|took .* ms  \(compiled\)"
```

Any `Module hash changed, recompiling: <module>` line for a `triwarp.kernels.*` or `map_*` module is
the defect. A *second* line for the same module in one run means the chain is still forking. **This is
the first thing to inspect because the alternative reading — "this test is inherently slow, cap its
input or delete it" — removes coverage to work around a fixable compile.**

**The second candidate is host-side per-element Python, and it is a distinct class** — not a rebuild
(the GPU is idle for both, so that tell does not separate them) and not launch overhead (those are
milliseconds, not minutes). One benchmark ran over 20 minutes without finishing at 0 % GPU because a
`validate=True` path built Python `set` comprehensions over a `.numpy()`-read array — one interpreter
iteration per mesh edge, invisible on ordinary meshes and catastrophic on the largest scan mesh.
**When a benchmark group stalls, grep the triwarp function it times for a Python `for` / `set` /
comprehension over anything derived from `faces` or `vertices`** — a `.numpy()` feeding a
comprehension is the signature. The fix is a device scan, **not a smaller benchmark mesh**: capping
the mesh hides the defect, which is what the largest row exists to prevent.

### 15.2 Attribute against one number

A projection built by subtracting two measurements of *different* things is a hypothesis, and in this
repo it has been optimistic by 3-10x every time. The one projection that held came from a single
directly-attributed number; three others that came in far under a rough subtraction-based estimate
each mixed up two different quantities — a fallback pass's cost estimated from two different
per-span metrics rather than the pass itself; an amortized-cost claim read from a warm,
unrepresentative solve rather than the real per-iteration cost; and a memset count that couldn't
actually be removed because it was initializing kernel inputs, not padding an allocation.

**Price the candidate directly** — time the exact call in isolation, or count its launches and
multiply. Two specific traps: a *warm* repeat of a stage measures a different regime than the cold
one inside the real call (a CG solve especially), and `wp.empty` vs `wp.zeros` differ in what they
actually remove, so "remove the memset" and "remove the allocation" are different claims.

**And price it at more than one size, because the *sign of the trend* is the decision.** Two items in
one pass had nearly the same share at the small end and opposite verdicts, which only a second size
revealed: one readback's share of its call *grows* with mesh size, so the fix is worth more the more
it matters and it landed; a different readback's share *falls*, which §9 calls a decline.

**The rule fails in *both* directions, and the pessimistic direction is the one that quietly throws
away real wins.** The failure above is a subtraction that flatters a candidate; the mirror is
measuring a candidate *through* a call it barely occupies, where the enclosing noise becomes the
verdict. A `heat` kernel fusion was declined on exactly that: whole-call A/B gave 1.02x then 1.006x —
read as §9's falling share — and a 25-row harness sweep gave 0.987-1.041x, whose sub-1.0 cells read
as regressions. Priced directly, the fused region is **1.04-2.65x at every size and never slower**;
it is 0.14 % of the call that reported it worst. **Before reading a ratio near 1.0 as a verdict,
compute the candidate's share of what you measured** — under a few percent, the measurement cannot
see it and the honest next step is to isolate the region, not to declare a decline. §14.10 has the
full case.

**A loop with a convergence break reports the break point, not the change.** An
`icp_point_to_plane` hoist read **1.86x** end to end and was worth **1.02-1.06x**: the two arms
stopped at different iterations, and nothing else about the apparent saving was real. Two directly
measured maps at ~10 µs each on a ~200 µs iteration was the whole of it. **Pin the iteration count
before timing an iterative solver** — `threshold=0.0` here — or the ratio is fiction. The same call
is also not bit-reproducible once converged (the point-to-plane normal equations accumulate through
`float32` atomics), so a value gate on it has to compare a *trajectory* and expect the plateau to
wobble.

### 15.3 Attribute at the benchmarked operating point

**A profiled share is a share at one point on the parameter axis, and it can reverse an
optimization's sign.** Attributing a blue-noise sampler at its natural, dense-sampling parameter
found one kernel dominating device time; inverting it (running one thread per *accepted* point
instead of per *alive* point) was a large win there — but at the actual benchmarked, sparser radius
the same inversion was a real *loss*, because far fewer points get accepted per round and the
inverted kernel starves the device instead of using it. Reverted. **Grep the benchmark for how it
derives its parameter and profile at *that* value** — §9's interleave rule doesn't catch this, since
both arms were timed correctly, just at the wrong radius.

**The same rule applies to the *fixture*, not just the parameter.** A `mesh_to_mesh_distance` fix was
first evaluated against a heavily-interpenetrating pair of copies, where the cost it targeted was a
small and shrinking share of a call dominated by something else — reading as a decline. The actual
benchmark fixture places the copies disjoint side by side, where that same cost is a much larger,
real share and the fix is a genuine win. **Read the fixture, not just the call.**

### 15.4 Benchmark-harness hazards

- **`benchmarks/test_meshes.py` is a real gate and the default `pytest` run does not collect it.**
  It self-checks the registry against a topology table every feature mesh must match, and a mesh
  registered without its matching row fails there with a bare `KeyError` while the full suite,
  `basedpyright`, `zensical build --strict` and `tests.parity` all stay green. **After touching
  `benchmarks/meshes.py`, run `pytest benchmarks/test_meshes.py`** as a fifth gate.
- **`--benchmark-json` is written at session end, so one pathological row costs the whole module.**
  A newly added reference library lost several modules this way — a size cap that governed every
  case except the one it existed for, a cap check that was unreachable dead code, and an uncapped
  row that starved a later triwarp row of device memory entirely. **General rule: put a
  `skip_larger_than`-style cap first in every group, before any library-specific branch; make sure
  it actually covers meshes outside the reference's own registry; free any foreign library's device
  memory in its teardown, outside the timed region; and put a bounded per-module wall-clock cap in
  the runner** so one bad row can't silently blank out the rest of a module's results.
- **Never benchmark Poisson surface reconstruction on very large point clouds with a reference
  library that doesn't scale to them** — one run went for over 90 minutes without completing a
  single row and had to be killed, blocking every module after it. **When a benchmark module's wall
  clock looks wrong, get the per-library split before trimming anything triwarp does** — the slow
  side is often the reference, not the code under test.
- **A `benchmarks/` harness number and an isolated hand-probe number for the identical call are not
  directly comparable** (the harness syncs and cold-pools differently around the timed region) —
  **compare probe to probe or harness to harness, never across, and label which kind a quoted number
  is.** Comparing across the two once nearly produced a false regression report.
- **A median at a low sample count (`rounds=3`) can be unrepresentative of its own samples** — a
  one-off cost (the shape of a Warp module load) landing in two of three samples can inflate a
  median far above the floor while the floor itself never moved. **Run the aggregate script's
  "suspect" check and read it before trusting a loss table** — it flags any cell whose median
  exceeds its own minimum by a wide margin, on both triwarp's and a reference's own cells (an
  inflated *reference* median flatters a triwarp "win" the same way). A flagged cell should be
  re-measured on the next round before anything is built against it.

### 15.5 A plan item may be refuted by its own target

In one plan round, **all three items worked were refuted, and in every case the refutation was
already written in the file the item proposed changing** — a counterexample named in the constant's
own comment, a scope mismatch the benchmark's own docstring already anticipated, and a threshold
whose tuning basis the plan misdiagnosed by looking at the wrong stage.

**A plan is written by reading loss tables and diffs; a measured decline is prose and lives next to
the code, so a table-driven pass systematically cannot see it.** Before measuring a plan item, grep
the target function, its constant's comment *and its benchmark docstring* for a number.

**And when a "nothing separates these" note exists, check which *kind* of quantity it ruled out
before accepting it.** Several attempts at a predictor tried different properties of the *solve*
(size, iteration count, convergence-rate extrapolation) and all failed the same way; the property
that actually separates the cases was a property of the *operator*, never tried (§16.8).

**Two refutations from the host-cost sweep, both cheap to re-derive and both wrong-looking-right:**

- **Narrowing edge/row keys to 32 bits to halve the radix sort.** `radix_sort_pairs` really is
  1.3-4.1x cheaper for `int32` than `int64` in isolation. End to end `unique_1d(return_inverse=True)`
  on `uint32` keys measures **0.91-1.01x** against `uint64` — no better, sometimes worse — because
  `uint32` is outside `_unique_hash`'s native `(int32, int64)` set and picks up a `bitcast_to_int`
  copy, and because the call is host-bound anyway. The lever was never the key *width*; it was the
  reduction *around* the packing (§16.4).
- **Replacing `map_sorted_inverse`'s binary search with a hash-slot lookup.** Recording each
  element's table slot at insert time and resolving the inverse as `inv_perm[scan_pos[slot] - 1]`
  trades `log2(n_unique)` dependent probes for three. Stage-profiled, `map_sorted_inverse` is **under
  5 %** of `unique_1d` best case, against an extra `n`-sized `int32` buffer and a wider
  `hash_insert`. Not built. **Stage-profile before optimising a stage**; the obvious suspect here is
  5 % and the radix sort's *host floor* is a bigger share of it.

### 15.6 A/B without `git stash`

**Do not `git stash` to measure current code against pre-change code.** The working tree is shared —
a stash cycle can sweep up in-flight edits arriving from outside the session while it runs, reverting
them on disk for a window in which a save would conflict.

Use a detached worktree, which never touches the working tree:

```bash
git worktree add -q --detach $SCRATCH/baseline HEAD
```

**The catch that makes it non-obvious:** `uv sync` installs triwarp through a *MetaPathFinder* that
outranks both `sys.path` and `PYTHONPATH`, so `PYTHONPATH=$SCRATCH/baseline python probe.py` still
silently imports the working tree. Drop the finder first:

```python
sys.meta_path = [f for f in sys.meta_path if "editable" not in getattr(f, "__module__", "")]
sys.path.insert(0, BASELINE)
```

**A `sed`-based in-place sweep of a tuning constant is the same hazard as `git stash`, and it's
worse because it looks harmless** — it mutates a tree another process may be running tests against.
A constant baked into a kernel can't be swept within one process anyway (Warp fixes it at codegen),
which is exactly the case that tempts the in-place edit; put the value in a detached worktree instead
and drive it with the main venv's interpreter.

**And check whether the box is yours before taking any clock reading at all.** `nvidia-smi`'s
utilization and a process listing cost nothing and are the difference between a measurement and a
fiction — a sweep once read a 4x swing at *fixed* parameters purely because foreign processes held
the GPU busy during part of it. Two rules follow. **Interleave A and B inside one process where the
change permits it** — coarser-grained interleaving is too slow to track noise on a second-scale
timescale. And **when the box is not quiet, measure a quantity that is not a clock**: iteration
counts, launch counts, level counts, candidate counts, `nnz`, and whether a reference mutated its
input are all deterministic, and can settle a question a noisy clock can't.

**But a failing `nvidia-smi` is not evidence that CUDA is unusable — probe the driver through
Warp.** NVML is a separate userspace library from the CUDA driver API Warp calls, so a host whose
`nvidia-smi` dies with `Failed to initialize NVML: Driver/library version mismatch` can still
report the device from `wp.init()` and launch kernels correctly, with the full CUDA suite green in
that state. Cost of not knowing this: one review pass recorded itself as CPU-only and wrote off its
own CUDA evidence. So `nvidia-smi` is the *box-is-quiet* check above and nothing more; availability
is `wp.get_cuda_device_count()` plus one real launch.

**torch reports the same mismatch as a test warning, and it is the suite's only one on a CUDA
run.** `torch.cuda.__init__` calls `_raw_device_count_nvml()` and warns `UserWarning: Can't
initialize NVML` when that library is the mismatched one, so a CUDA run of `tests/` ends in
`2 warnings` where the CPU run ends in none. It is the box, not the tree: nothing in triwarp or in
the reference stack raises it, no code change silences it, and reloading the NVIDIA kernel modules
(or rebooting) is the whole remedy. Do not filter it — it is the one standing signal that this
host's NVML is out of step.

**Verify with `print(triwarp.__file__)` before trusting a single number.** Do **not** `uv run` from
inside the worktree — it resolves that copy as its own project and builds a second virtualenv; use
`.venv/bin/python` directly.

### 15.7 Timing hygiene

- **Interleave A and B in one loop and report the `min` alongside the median.** Timing all of A then
  all of B lets GPU clock state decide the winner — one sweep produced non-monotonic ratios and a
  recurring artifact that reversed into a clean monotonic trend once interleaved under one clock
  state with the GPU pre-warmed.
- **Saved baselines drift ±10% (±30% under 100 µs) between sessions.** Flagged deltas in that range
  have appeared on benchmarks whose code hadn't changed at all; re-running both arms back-to-back
  showed the true delta was much smaller, with the sign flipping between reps.
- **A reference library's own column is the control that licenses a cross-session comparison.** A
  stale benchmark table once carried findings built on numbers that had drifted since — but because
  the *reference's* column was unchanged within noise across the two sessions, that reference served
  as a control proving the drift was real and belonged to triwarp's side. **Re-run the *whole group*
  including its reference rows**, and a table that stale is worth rewriting rather than annotating,
  since its *conclusions* mislead more than its numbers.
- **Do not `wp.synchronize_device` around each launch** when timing a microsecond kernel — that
  measures sync latency, not the kernel, and can read wildly different numbers for the identical
  kernel depending on launch count. Batch K launches, sync once, divide.
- **The device/wall timer's per-launch synchronization inflates *wall* badly** — one case was off
  by more than 10x. Take launch counts and device totals from
  `wp.timing_begin(cuda_filter=wp.TIMING_KERNEL | wp.TIMING_MEMSET, synchronize=True)`, and wall time
  from a separate un-instrumented loop.

- **A/B one pytest process per module, never several modules in one process.** An A/B of fifteen
  benchmark modules in one invocation put ~24 cells under 0.93x -- `face_adjacency_angles[dragon]`
  0.47x in two separate runs -- and every one was level with its module run alone (§16.23): the
  arm's timing depended on the allocator and cache state the preceding modules left. The round
  sweep's `run.sh` is one process per module for this reason; an ad hoc A/B must be too.

### 15.8 Probe-process contamination

**A loop-over-configurations probe script is not the same experiment as the code under test.** One
process looping over several cloud sizes and depths produced what looked exactly like a
nondeterministic backend bug (the same configuration returning correct output on one run and an
error on the next, sandwiched between succeeding neighbors) — but running each configuration in its
own fresh process gave the correct output every time, identical to CUDA, matching what the real test
suite already showed. **The "defect" was the probe's own cross-configuration state leaking, not the
product.**

**Before attributing a failure seen in a multi-config probe to the product, re-run the single
failing configuration in a fresh process**, and prefer running the real tests over re-deriving them
in an ad hoc script. **Corollary for the reverse direction:** a probe that *passes* in one process
says nothing about a suite that runs hundreds of allocations before it — holding several very large
structures alive at once in one process has separately produced a CUDA allocator error that was the
probe's own footprint, not a reproduced defect.

### 15.9 Decide on the harness number, not the isolated one

`check_every=0` for the heat-family CG read 2.1-2.3x in an isolated probe and 1.05-1.23x in the
harness, with **0.94x** on the `saddle_graded` conditioning rows (§14.6). Same shape as §15.4's
harness-versus-probe factor: **decide on the harness number.**

### 15.10 `wp.timing_begin` is blind to graph-replayed kernels, so a captured function reads as ~100 % host

**This invalidates the device/wall split for every function that graph-captures**, and it fails in
the most expensive direction: a device-bound function reads as host-bound, which points the next
optimization at launch elimination when the kernels are the cost. Measured directly: 20 loose
`wp.launch` calls report 20 kernels to `timing_begin`; the identical 20 calls replayed from inside a
capture report **zero** kernels and zero time.

**The reach is much wider than triwarp's own handful of explicit capture sites**, because
`warp.optim.linear`'s solvers capture their iteration by default. So every triwarp CG solve — heat,
parametrization, smoothing, `min_quad_with_fixed`, `solve_spd*` — has its dominant kernels hidden
from this measurement, and at least three functions were attributed backwards this way.

**How to measure it correctly.** Re-run with capture disabled and read `timing_begin` there; the
kernel *set* is unchanged, so the device total transfers back to the captured run (only the wall
does not — the uncaptured wall is several times higher, which is what the capture is worth).
`warp.optim.linear` takes `use_cuda_graph=False`; triwarp's own `wp.capture_while` sites fall back
to direct execution when `wp.is_conditional_graph_supported` returns `False`, so monkeypatching that
is the lever. **Both arms must be instrumented identically** (§9) — the uncaptured arm is the
measurement, not the baseline.

**The tell that this was wrong all along was already in the tree**: a docstring had concluded the
call was device-bound from a solver-tolerance sweep, not from `timing_begin` — and that sweep was
right. **When a device/wall split contradicts a tolerance or input-size sweep on the same function,
trust the sweep**: it cannot be fooled by where the kernels were issued from.

**Two second-order traps in the same family, both of which make host time look larger than it is:**
a `.numpy()` readback's wall time is the queue depth in front of it (§14.6), not its own cost; and
once enough launches are pending, the driver's launch queue fills and `wp.launch` itself blocks —
a device-bound signature that can look like marshalling cost if read naively.

### 15.11 Census the host calls by monkeypatching Warp, not by grepping

A whole class of per-call cost is invisible to a static scan, because it depends on *runtime types*
(is this value a `wp.int32` or an `int`?) and on *how often a line runs* (once per call, or once per
iteration?). Two monkeypatch censuses answer both, and they are the tool §13.1's dispatch table and
§16's per-call attributions were derived with. Neither needs a benchmark.

**Census 1 — Python-scope builtin dispatch.** Patch `warp._src.context.Function.__call__`, walk out
of `/warp/_src/` frames to the first triwarp frame, and count by `file:line`. Record the operator
dunder on the way out (the first `/warp/_src/types.py` frame's `co_name`): `via=__mul__` is
*arithmetic on a Warp-typed value* and a defect, `via=None` is an explicit `wp.length(...)` /
`wp.cross(...)` call and a judgement call. Run it under `pytest tests -q`.

**Census 2 — host-call counts by call site.** Patch `wp.array.__getitem__`, `wp.array.numpy`,
`wp.zeros`, `wp.empty`, `warp._src.utils.map` and attribute the same way. Suite-wide over `tests/`
on CUDA this measured **216 911 `wp.empty`, 52 816 slices, 44 755 readbacks, 34 433 `wp.zeros` and
6 566 `wp.map`** — about 2.4 s of a 68 s run at §13.1's prices.

**Read the *slope*, not the count.** A suite-wide total cannot tell a line that runs once per call
from one that runs once per iteration. Run the same function at two iteration counts and difference
the per-site counts: that isolates the loop-invariant repeats, and it is what found
`linalg._dot_finalize`'s `self._dots[1]` at **2.00 slices per CG iteration** and
`smoothing.filter_implicit_fairing`'s column views at 6 per pass. It is also what **refuted** several
statically-derived claims in the same round — `filter_taubin`, `filter_humphrey` and the blue-noise
rounds showed *no* per-iteration growth, having been fused already by §14.10.

**Four caveats, every one of which produced a wrong reading before it was understood:**

- **A census only sees what runs, so it is a two-device job** (§7.2). The CUDA pass could not see
  `graph.shortest_path_envelope`'s slice at all, because it sits inside `if not device.is_cuda:`;
  the CPU pass found it immediately.
- **It can misattribute a *Warp-internal* call to the triwarp caller.** The frame walk stops at the
  first non-Warp frame, so a builtin that Warp itself calls inside `wp.MarchingCubes.extract_*`
  is charged to the `triwarp/levelset.py` line that called it. **Read the source line before
  believing a hit.**
- **Vector arithmetic is invisible to census 1** (§13.1): `vec_t.__sub__` goes through `_binary_op`,
  not `Function.__call__`. A clean census does not mean no Warp-typed arithmetic.
- **A capture hides a loop from census 2 the way it hides kernels from `wp.timing_begin`**
  (§15.10). With `check_every == 0` on CUDA, `_BatchedCg._iteration` runs *once*, at graph-record
  time, so its per-site counts are per *solve*; the same code on the host-check path runs per
  iteration. Take the slope on the path you mean to price.

---

## 16. triwarp component status

Open defects, refuted plans, and the rules the optimization rounds left behind, by area. **Check
here before opening work on any of these.** What *shipped* is in the code; what is recorded here is
what a reader would otherwise re-derive.

### 16.0 The launch-resolution pass, and what it says about where to look next

Every generic-kernel launch site goes through a dtype-keyed table of concrete kernels (§2.5, §13.1).
Three method points that generalise past it:

- **Take the census from the runtime, not the parser**, for any property Warp computes rather than
  the source spells. A *factory* can leave a `dtype` parameter generic by default with nothing in
  the source text saying so; `[k for k, v in wp.get_module(name).kernels.items() if v.is_generic]`
  finds those where an AST scan of annotations does not.
- **A written decline is not a landed one.** A docstring documented "the generic form is declined
  here" for years while a third of its own instantiations took the generic default, because a
  factory parameter had one. §9's "the decline may already be written there" has a converse: grep
  for the sites that should have obeyed one and did not.
- **A probe that instruments the thing it measures must carry a do-nothing control arm.**
  Monkeypatching `wp.launch` to substitute concrete kernels reported *losses* that were artifacts of
  the patch: it left non-array generic parameters generic and added a Python frame to one arm only.

### 16.1 Where triwarp's benchmark losses actually are

**The whole mid-level surface is host-bound.** 44 public functions timed over a 256x face range came
out flat within 1.25x and none above 3x; `grouping.unique_1d` corroborates independently at 82 %
host. Two consequences, and they set the shape of every optimization in this package:

- **There is usually no kernel to make faster.** The currency is the *count* of Warp API calls and
  §13.1 is the price list — triwarp's own Python is a rounding error beside it. Micro-optimising
  wrapper code is not a lever; removing Warp calls is.
- **Flatness across the mesh size is the measurement to take first**, before any profiler. Two
  timings, no instrumentation, and it cannot be fooled by graph capture the way a device/wall split
  can (§15.10).

Most of the biggest losses are 92-99 % host-side launch and allocation cost, and almost everything
in the 0.3-20 ms band is 66-97 % host at high launch counts; the exceptions (`ambient_occlusion`,
`lscm`, `query_nearest_bvh_k1`) are genuinely device-bound. **Tiling cannot touch a launch floor;
only launch *elimination* can**, and reading a benchmark table without the device/wall split
misdiagnoses most of this band — first hypotheses that assumed kernel or algorithm cost were wrong
more often than not.

!!! warning "Only valid for a function that does **not** graph-capture"
    `wp.timing_begin` reports zero kernels for graph-replayed work (§15.10), and
    `warp.optim.linear` captures its solver iteration by default — so this method reads any
    CG-backed or `wp.capture_while`-driven function as ~100 % host whatever it really is. **Check
    for capture before trusting a host share, and re-derive one taken before a capture landed.**

The rows that *are* device-bound are so because the grid is too narrow to fill the machine, not
because a kernel is inefficient. One single-kernel outlier is unrelated to launch count:
`linalg.assemble_interior_system` spends most of its device time in one
`_bsr_accumulate_triplet_values`, because it routes an already-row-sorted, already-deduplicated CSR
through `bsr_from_triplets`.

### 16.2 `import triwarp`

`triwarp/__init__.py` resolves each submodule (and `Trimesh`, a class, as a special case) through a
PEP 562 `__getattr__` and caches it into the module namespace. Cause: §12.6 — `@wp.kernel` builds an
`Adjoint` at import time for every decorated kernel, and importing a submodule does not avoid it
because Python imports the parent package first.

**The guarding test must run `import triwarp` in a subprocess and assert zero kernel modules are
pulled in** — in-process the answer is always "all of them", because by the time a test runs the
session already imported what it needed. Do not "simplify" it to an in-process check. Two things
the laziness deliberately does not change: overload registration still runs before the first launch
through its module, and nothing calls `wp.load_module` / `wp.force_load` at import.

### 16.3 `reconstruction`

- **`screened_poisson`'s `dense` solve is over a `2^depth`-cubed node grid whatever the cloud size**,
  so each level costs ~8x and the CPU test depth is one lower than CUDA's. **Error is not monotone
  in depth** — past some depth the octree resolves sampling noise rather than the surface — so a
  test must never assert "finer depth reduces error".
- **OPEN: `screened_poisson(point_weight=0.0)` returns a handful of zero-area triangles, every run**
  — the only one of the four reconstruction entry points with no degenerate-face cleanup pass. Found
  through a reference library's warning about it; §7.7 has the general technique.
- **OPEN: `repair.make_winding_consistent` seeds each connected component from an arbitrary face**,
  so per-component winding is not reproducible even though the unoriented triangle set is.
- **`ball_pivoting` design choices that are counter to the obvious guess — do not re-try them.**
  `wp.capture_while` is *slower* than a batched host loop for this loop shape (§14.3); the hash-grid
  cell width must be the *ball* radius, not the wider pivot neighbourhood (a wider cell makes the
  far more frequent empty-ball query enumerate many more points than it needs); and the persistent
  front is what fixed the old watertightness limitation, so the "overlapping sheets" caveat belongs
  to the per-wave rebuild and must not be reinstated.
- **When making an order-dependent algorithm deterministic, check whether the replacement order is
  fixed across *rounds*, not just within one.** BPA's nondeterminism was integer arrival order — a
  proposal slot handed out by one atomic reused as the *priority* for a second — and a globally
  fixed key alone *starves* the front, because the same proposals win the same contested vertices
  every wave. The key is salted with a hash of the wave counter.
- **No uniformly-sampled closed sphere can test that**: its exact Euler triangulation leaves wave
  order nothing to decide, so only an irregular-spacing fixture exposes the nondeterminism at all.
- **`ball_pivoting`'s default (`radius=0`) auto-radius is nondeterministic**, and the reproducibility
  claim is scoped to an explicit `radius=` deliberately. Do not "fix" the underlying reduction
  without a caller that needs it (§4.2).
- **When the memory tools come back clean on a memory-shaped symptom, instrument the control flow
  instead** — they perturb the very timing the defect's schedule depends on. BPA's intermittent
  `CUDA error 700` was a host-side front-buffer swap desync, not the allocator or a lifetime bug;
  several plausible readings were symptoms of it, which is why one fix closed all of them.
- **Slab-chunked marching cubes is not viable.** `wp.MarchingCubes` is crack-free only *within* one
  grid — its per-cell triangulation is not consistent across independent invocations — so welding
  independently-computed slabs leaves non-manifold seams.
- **`warp.fem` gotchas, each producing a plausible wrong answer rather than an error:** an
  `ImplicitField` func must have no return annotation; `allocate_by_voxels` is voxel-*centered*, so
  the extraction lattice needs a half-voxel translation or it produces a spurious surface component;
  solve in *index space* so the screening-versus-stiffness balance matches the dense calibration;
  and a point-source weak form rings when cells are much smaller than the sample spacing, capped by
  limiting the grid depth rather than rounding it.

### 16.4 `remesh`, `repair`, `creation`, `bounds`

- **A green parity test is evidence only if its fixture can express the difference.** The remesh
  concentration test ran on a uniform icosphere, where area weights equal uniform ones, so the smooth
  stage's whole job was invisible and both arms measured bit-identically. §7.4's vacuity rule applied
  to the *input*: run the new code and the old against the parity fixture and diff the statistic.
  It is now parametrized over a uniform sphere **and** a graded patch, and disabling any one of
  split / collapse / smooth fails it.
- **A graded regular grid is the pathological remesh input**: valence-perfect, already Delaunay, and
  a fixed point of the unweighted Laplacian, so three of the five stages are blind to its anisotropy
  by construction. A suite that only runs the remesher on clean closed icospheres shares that blind
  spot.
- **A tangential step to a convex combination of the one ring cannot fold a *convex* vertex link**,
  so every well-shaped fixture is unable to reach the smooth pass's fold veto and a test built on one
  asserts nothing. The guard's test needs a deliberately non-convex link. The weighting itself is
  pinned by a hand-computable NumPy oracle, since no reference exposes a single relaxation step.
- **The isotropic collapse and the quadric collapse share one fold veto** — a duplicated *decision
  rule* diverging is §2.4's hazard, and this was exactly that gap. **The adjacency build the veto
  needs is free**: a rejected collapse removes downstream work that pays for it.
- **An independent-set kernel needs a test on how many winners a round produces**, not only on the
  validity of the ones it commits. Locking on the raw edge index is spatially monotone, so nearly
  every round had one winner among tens of thousands of candidates — silently correct-looking,
  because nothing asserted a collapse count.
- **A min-key parallel independent set needs its key to be both spatially incoherent *and*
  injective** — two separate properties. A hash alone lets two candidates collide and both commit,
  corrupting the mesh; the fix packs the hash and the raw index into separate halves of a 64-bit key.
  A correctness guard bolted onto only one of several callers is the tell that the key, not the
  caller, is wrong.
- **A collapse anti-oscillation test must walk *both* rings** when the placement is free, since that
  moves the survivor too and every edge from its own neighbours to the new position is otherwise
  untested.
- **`_valence_flip_pass` reads valence off the sorted key buffer its own edge rebuild already
  produced**, not from a second `edges_unique`. **Trap: the natural "runs of exactly two" marker
  undercounts** — it flags manifold-interior edges only and silently drops every boundary edge; the
  marker that matches `edges_unique`'s row set is the any-length run start.
- **`isotropic_remesh` is not byte-gateable** — atomic-order drift in normal and ring accumulation
  tips split/collapse decisions and changes even the face count. **To gate a change it merely
  contains, gate the exactly-reproducible contained stage** (e.g. `_valence_flip_pass`) rather than
  loosening the comparison on the whole function.
- **Edge-length equilibrium is ~1.0x target only when the target is a "nice" ratio of the input
  edge** (midpoint-split quantization); coarser-than-input targets plateau lower.
- **A count the producing kernel already knows is a conditional `wp.atomic_add`, not a reduction
  over its output mask** (§13.2 — contention scales with the hits, not the launch).
- **Hoisting per-pass scratch is a loss when the loop usually runs once.** Hoist against the
  *expected* pass count, not the maximum.
- **A whole-buffer `fill_` / `zero_` whose length equals the `dim` of an *adjacent* launch that does
  not read it belongs inside that launch.** The counter-case is `sample._dart_throw_blue_noise`,
  whose buffers are cell-sized while its launch is over a shrinking alive count — no same-`dim`
  launch to fold into.
- **`creation.parametric_surface`'s lattice has a size gate** (`_PARAMETRIC_LATTICE_DEVICE_FROM`)
  with **both** implementations kept: the device path has a flat launch-and-readback floor and is a
  2.4x loss below it, the host path is quadratic in the resolution. Same shape as §14.7's
  `_DOWNSAMPLE_DOUBLING_FROM`. **Every resolution the suite uses is below the gate**, so its test
  monkeypatches the threshold both ways and compares bit-for-bit — otherwise the device path is
  unreachable in an ordinary run.
    - Its pole anchor *is* statically derivable (always the corner the pole's edge implies) — but
      prove it with a probe over every spec and resolution rather than reasoning it. Two orderings
      are equally easy to get wrong and are called out at the kernel: the u-seam
      re-canonicalisation after a v-twist is **unmasked**, and all four pole masks are snapshots of
      the post-wrap state taken *before* any collapse.
- **A numpy reduction over a short trailing axis is a pathological shape, not a cheap one.**
  `sides.prod(axis=1)` on an `(n, 3)` array is an order of magnitude dearer than
  `s[:,0]*s[:,1]*s[:,2]`, and a strided difference on a stride-24 view ten times its contiguous
  cost. Worth checking before pricing any host-side table op by its element count.
- **A host helper that reproduces a kernel's arithmetic is §2.4's duplicated decision rule across
  the host/device boundary.** `bounds`' `_spiral_frames` reproduced the candidate-axis spiral in
  numpy and was never bit-identical — the host narrowed after building the matrix, the kernel
  narrows the quaternion first — so the refinement descended from a frame fractionally different
  from the one that had been scored.
- **A search whose answer can tie must be tested on the *achieved objective*, not the frame it
  returns.** Two of 105 oriented-box configurations select a different frame under a 1-ulp reduction
  difference, because a rotationally symmetric needle has three candidates at the exact minimum.
- **A guard whose only failure mode is an out-of-bounds read needs the check, not a test.** The
  oriented-box chain padding is not gated by any value test and cannot be — a chain seeded from a bad
  frame simply loses the final argmin — so the kernel clamps the index at the read (§12.1:
  range-check at the access, not downstream of it).
- **Declined: a global farthest-first oriented-box seed.** Less than half the faithful walk's cost
  and better box volumes on average, but worse on one shape by enough to eat half the margin
  `test_oriented_bounding_box_refinement_is_monotone` documents. It needs its own evaluation across
  `MESHES` rather than riding on a change that alters no answer.
- **When specialising a general routine, call its *predicate* rather than reimplementing it — the
  specialisation is the layout, not the rule.** `cone` / `cylinder` / `revolve_uniform` are
  closed-form kernels that still ask `revolve`'s own `_revolve_kept_template` which triangles
  survive, which is what makes them byte-identical at the degenerate section counts and inherits its
  absolute area tolerance.
- **A fast path is *checked against the general engine*, not asserted, and the fallback must be
  measured live.** `_revolve_regular` returns `None` whenever the surviving triangles are not the
  regular pattern; 10 of 27 probed configurations take that branch.
    - Its profile cap (`_REVOLVE_REGULAR_MAX_PROFILE`) is set by an asymmetry: the win when the
      screen succeeds is flat, the loss when it declines grows with the profile. Above the cap every
      shape probed declined, so the screen was pure overhead on top of the general engine.
    - On a decline the template is computed twice. Removing that needs it threaded through
      `revolve`, whose public signature may not name an `np.ndarray` (§3.8) — not worth it.
- **REFUTED — reducing a concatenation of two clouds instead of seeding one accumulator twice**
  (`bounds.enclosing_diagonal`). A loss at every size from 1 000 to 4 000 000 points: the union is an
  allocation and a copy of all the data to remove one `wp.launch`, which is flat in its `dim`
  (§13.1). The *first*, non-interleaved pass of that A/B reported concatenation **winning** — §15.7
  with a number on it.
    - The negated upper corner in that reduction is orthogonal and looks connected: it exists so one
      `wp.full(6, inf)` seeds both ends and every update is an `atomic_min`. Unpacking it would need
      the two-value seed §3.3 forbids and would fork a packing four readers share.
    - Both launches take their `dim` from `kernel_reduce.chunks_1d`, the §13.2 counterpart to
      `blocks_1d` for the *unfolded* `TILE_1D` kernels; `blocks_1d` is wrong for them by a factor of
      `TILES_PER_BLOCK_1D`.
- **Producer-then-reduce is a tree-wide pattern with three mechanisms, and the second is easy to
  miss.** Fold the reduction into its producer; express a *closed* polyline by **wrapping the index**
  rather than by a whole-buffer copy (once the reduction is fused, that copy *is* the call); and
  where the readback must stay because a Python loop branches on it, drop the reduction around it.
  An `ast` scan for "a map/launch writes a buffer, a `tw.reduce.*` reduces it within a few
  statements" finds ~33 sites; most spend 1-4 % of the call there and are declined.
    - **A lane-strided fold plus `wp.tile_sum` is not the sequential accumulation `reduce.sum`
      performs below `TILE_1D`**, and it is the *more* accurate arm — a tree halves the depth over
      which rounding compounds. But it breaks any test asserting bit-identity with a reference's
      sequential sum: **check for an exact comparison before fusing a reduction**, and if the trade
      is taken, rewrite the test's reasoning rather than just loosening its number.
    - A scan keyed on line proximity over-counts: three reductions over one array can be three
      *mutually exclusive branches*. Read the function before believing the hit.
- **`edges_unique` is a host-bound substrate under ~57 call sites, and three things keep it there.**
  The deduplicated rows are already recoverable from the packed keys (`array.unpack_edge_key`), so no
  scatter-and-gather is needed; `validate` is a keyword defaulting `True` with every internal caller
  passing `False` (`_device.require_valid_faces`' own docstring prescribes exactly that split, and
  every downstream per-face kernel already trusts the connectivity); and `index_bound(X)` followed by
  a validating hash of `X` reduces the same array twice, so `array.index_bound` takes both ends out
  of one `reduce.minmax` via `require_non_negative=`.
    - **A grep for `validate=False` on one line undercounts** — half the converted call sites wrap
      across lines.
- **A two-column index row needs no bound at all.** `hash_indices_rows` packs
  `sum(digit[i] * radix ** i)`, so *any* radix above every entry is injective, and every `int32`
  reinterpreted as `uint32` fits two columns inside a `uint64` (`constants.INDEX_RADIX_PAIR`). The
  packing stays monotone lexicographic, so sorted row order is unchanged. **Before assuming a bound
  is free to widen, ask what else it is**: in `validation.is_vertex_manifold` it is the *length* of
  the per-vertex mask, so widening it would change the answer on a mesh with trailing spares.
- **A heavy per-element integrand wants a chunk width chosen on the host, not the reduce family's
  constant.** `measures._moment_chunk_faces` doubles the chunk until the grid is no wider than
  1 280 blocks; a single 1 024-element chunk starves the device at the large end and a narrow one at
  the small. **A runtime width measures identical to a `wp.constant` one**, so nothing is lost by
  choosing it on the host — worth knowing before baking any chunk constant into a kernel.
- **A function returning several small device values writes them into one buffer and reads it once**
  (§3.10) — `points.fit_plane` / `principal_axes`, `measures.moments`, `registration`'s scalar
  accumulator.
- **`creation.sphere_cap`'s ring inverse is the one subtlety**: a thread recovers its ring from its
  vertex index by solving `3 r^2 - 3 r + 1 <= v`, and the ring *starts* are `1 + 3 r (r - 1)`
  **except ring 0**, which is the lone apex at slot 0 rather than that formula's 1. Getting it wrong
  wound every apex triangle around its neighbour and was caught by a byte-identity gate and by
  nothing else — the mesh stayed watertight, correctly wound and the right size.

### 16.5 `array`, `graph`, `polyline`, `intersection`

- **The three searches in `kernels/array.py` are not interchangeable and picking wrong is silent.**
  A wrong-search bug on an argsort payload returns a *valid* index of the wrong element, so seeds
  land on neighbouring elements and a flood fill returns plausibly too much. **When a binary search
  feeds an argsort payload, verify one lookup by hand against NumPy.** And **when a fix does not
  change a symptom, suspect two causes** — a second, unrelated bug in the same function produced the
  identical symptom.
- **`side="right"` is poisoned by `NaN`** (§12.4), so `unique_1d(return_inverse=True)` was silently
  wrong on any float array containing one. It searches `side="left"` and falls back to the last slot
  only on the `NaN` branch. Found by *registering the float overloads* (§2.5): asking what dtypes the
  wrapper's dispatch reaches surfaced a dtype nothing had ever knowingly launched.
- **`array.isin` takes a caller-supplied `max_index`, not `assume_unique`** — the latter buys nothing
  here, since neither triwarp strategy dedups. **The bound must be range-guarded inside the kernel**:
  an over-tight bound is an out-of-bounds gather, i.e. §12.1's host-heap corruption on CPU, not
  merely a wrong answer.
- `flatnonzero` uses an inclusive scan plus a single tail read; `scatter_index_where` expects the
  inclusive scan and writes at `inclusive[i] - 1`.
- **`intersection._link_segments` is vectorized NumPy pointer doubling** (Wyllie, extended to open
  chains). **The device port was deliberately not taken**: most of the remaining cost is already one
  non-portable sort, and a device version would regress the common single-contour case.
- **A helper that densifies or indexes *the whole mesh* under a consumer whose answer is a small
  subset of it is the shape to look for.** The tell is a call whose cost is flat in the mesh size
  while the answer is not. `marching_triangles` packs its own sorted-vertex-pair key rather than
  taking a mesh-wide unique edge id; the trade has a crossover (the host `np.unique` grows with the
  *level set*) but levels out rather than losing, so no gate is needed — the same swap on a function
  whose answer is mesh-sized would be a loss.
    - **REFUTED — replacing that densification with a sorted join** (`argsort` the starts,
      `searchsorted` the ends). Slower at every size, because it needs **two sort-class passes where
      densifying needs one**: `np.searchsorted` of `n` into `n` is `n log n` *dependent,
      cache-missing* probes rather than a streaming sort, and `np.argsort` is several times
      `np.sort`. **Densifying is not overhead you pay to enable a lookup table — the dense labels
      make the join itself O(1) per element, which buys more than the search they avoid.**
- **Where the values are known-bounded indices, a mask is a cheaper spelling of `unique_1d`'s
  answer** — a zeroed count-sized mask, a membership scatter and `flatnonzero` return the same sorted
  distinct values, against a hash table, a compaction, a radix sort and two readbacks.
- **`boundary.boundary_loops` is a floor row, not an open one.** Its largest piece is
  `successor_cycles`' pointer-doubling loop, and **both mechanisms for removing it are already
  refuted**: a CUDA graph cannot be reused across calls because the round count varies (§14.3's
  record-and-replay-once), and a single persistent block is the shape §14.9 refuted for `graph.bfs`.
  Its gate must include `mobius` — the fixture that actually takes the non-orientable branch —
  without which it is vacuous (§7.4).

### 16.6 `proximity`, `metrics`, `neighbors`

- **`query_hashgrid_nearest`'s cost is *cubic* in how far `initial_radius` under-estimates the answer
  distance**, because that one scalar sets both the cell width and the search seed. The default
  estimator inverts the *target* cloud's density, which is wrong whenever the two clouds are
  displaced, and the cost is genuinely in the cell walk, not the linear-scan fallback.
- **Seed the *backward* search of a symmetric distance query from the forward half's own answer** —
  both directions share one distance scale, so no probe or subsample is needed.
  **`backend="bvh"` at those call sites is a loss at every size tested — do not re-propose it.**
- **REFUTED — seeding the *forward* pass from a query-prefix probe, gated on size.** Sweeping only
  mesh size makes it look like a clean gateable cliff; holding size fixed and varying the cloud
  separation shows the real variable is the ratio of answer distance to point spacing, and it is not
  even monotonic in that. **A sweep across one axis can look like it identifies the real threshold
  while only correlating with the true variable on one fixture — vary the fixture's own free
  parameters before trusting a gate.**
- **REFUTED — an automatic cell-width trigger for `neighbors._knn_cell_size`.** A brute-force probe
  over a cloud subsample *does* recover the missing displacement term (subtract the subsample's own
  spacing or the estimate over-widens 3-5x), and is worth 2.0-11.4x through the middle displacement
  band — but the probe alone doubles the on-surface call that is the benchmarked operating point,
  and at large displacement the wide cell is a 1.6-2.2x loss because the linear scan is by then the
  right algorithm. **The bar is a number: get the probe under ~0.05 ms and it ships.** Meanwhile the
  lever is the public `initial_radius=`.
    - **The block-cooperative lead is refuted twice over**: the grid walk cannot be lane-split at all
      (§12.2, no per-cell entry point), and the linear-scan fallback is *not* load-imbalanced where
      it costs — nearly every row takes the scan there, so the launch is uniformly expensive.
    - Method note: `knn_sorted_insert` binary-searches its row's **whole length**, so a probe handing
      it a row wider than `k` gets silent garbage that reads exactly like a triwarp defect. Confirm a
      census against an independent oracle before believing it.
- **`query_nearest`'s non-monotonic drop is `k >= 8` and hash-grid-specific** (the BVH backend is
  monotonic over the same sweep): once a row's true k-th distance exceeds the grid's widest search
  radius it falls back to an exact O(n) scan, and that tail grows with `n`. **Fixed in round 23,
  and the old advice (`backend="bvh"` at moderate `k`) is stale.** The scan was the grid's whole
  weakness: the sparse end of a density-graded cloud, a volumetric cloud's boundary shell (14 % of
  rows on a 20 k uniform cube), a LiDAR ground plane -- 0.07-0.54x against `bvh`. A row whose query
  sits on a point of the cloud (nearest found within `DEFER_NEAR = 0.5` initial radii) is now
  marked `DEFERRED_ROW` and finished by the BVH row kernel (`only_deferred=1`), for `k <= 64` at
  `n >= _KNN_DEFER_MIN_POINTS = 8192`. The grid then wins every self-query cloud measured, 1.1-3.6x
  (`dragon` 2-3x), and every in-tree self-query dropped `backend="bvh"`. The price is one readback
  per `k > 1` default query: 0.89-0.97x on uniform surface clouds. **Refuted: deferring off-cloud
  rows too** -- a small ball in the empty space inside a surface cloud overlaps many interior BVH
  nodes and cost more than the scan (`knn_far` 1.42 -> 3.04 ms on `bunny_decimated`). One measured
  loss is kept: `interpolate_from_points`' `k` path on queries one to two spacings off the
  cloud, 0.42x on `bunny_decimated`, against 1.6-6.6x on far (`grid_points`) queries -- and a mesh's
  own vertices near the cloud are a documented use too, so this is a real regression band.
  **REFUTED -- widening the deferral to cover it** (`r23x_defer_near.txt`, `DEFER_NEAR` swept 0.5 /
  1 / 2 / 4 / everything): the losing cells do not move (`bunny_decimated` sits under the 8 192
  gate at 8 171 points, and `bunny` at two spacings reads 5.7-6.6 ms at every value against
  `bvh`'s 1.34), and the far set gets worse, up to 2.1x (`bunny` far k = 30, 5.5 -> 11.6 ms). A
  hand-set `initial_radius` of 2 or 4 spacings does not help either. The cost is the grid walk
  itself: an off-surface row doubles its radius up to the widest walk before any finish runs, so
  no choice of finish can recover it -- a fix has to decide *before* the walk. Over the same 34
  cells the grid beats `bvh` in 24, by up to 3.6x (and 3-4x on every `dragon` and far set), so no
  backend dominates and `interpolate_from_points` keeps the default.
- **"It broke a bit-exactness test" and "it changed the answer" are different findings, and only the
  second justifies a revert.** The cheap unsorted adjacency builder was declined for years because
  its `wp.atomic_add` row order differs run to run; the ball's answer never changed at all (same
  counts, same sets), and the curvature moved ~1e-06 on a field of range 4, at ill-conditioned
  vertices the docstring already tells callers not to read. The fix was to make the rows canonical
  (`array.sort_segments`, one launch), not to abandon the builder. **Measure the magnitude before
  concluding, then look for the third option that keeps the gain.**
    - An **in-process monkeypatch of the builder overstated the win**: substituting a Python shim for
      the `BsrMatrix` in one arm only is §9's instrumented-probe hazard in its cheapest disguise.
      §15.6's detached worktree is the authority.
- **The per-row shell sort is O(d^1.3) on one thread**, so it is for a bounded-degree graph: a win to
  a hub degree of a few hundred and a large loss past it, where the answer is `edges_to_csr`. A shell
  sort rather than insertion because a segment's width is *data* — there is no host-side branch to
  escape to `segmented_sort_pairs` the way `array.sort_rows` has.
- **REFUTED — a streaming Givens QR for `curvature.principal_curvature`'s quadric fit.** Against the
  *exact rational* least-squares solution it is ~430 000x less order-sensitive and 2-4x slower, and
  exactly one vertex of 544 has a normal-equation error worth caring about — at 2 % of the field
  median, the regime the docstring already disclaims. The mechanism is the per-row cost: normal
  equations do one `wp.outer`, QR annihilates five entries each needing a **`float64` `sqrt`**. **The
  penalty grows with the neighbourhood size, which is backwards** — the large-radius call is both
  the expensive one and the one a robustness fix would be for.
- **REFUTED with it, and more instructive: symmetric Jacobi scaling of those normal equations buys
  nothing** — it is free (once per vertex, not per row) and measures a hair *worse*. That is the
  diagnosis, not a null result: the ill-conditioning is **genuine near-rank-deficiency of the
  neighbourhood** in the tangent plane, not a column-scaling imbalance, and the fit already divides
  its local coordinates by the ring radius.
- **A tie that is exact in float32 is not a tie in a float64 oracle, and an exact set compare then
  pins the adjacency's column order rather than the answer.** Where two candidates are bit-identical
  in float32 and one is backfilled, no correct tie-break exists at the precision the device works in.
- **`k`, not `n`, is what is still slow in the k-NN path** — insertion cost is super-linear in `k`
  because the candidate row lives in global memory and both the shift-insert and the reset touch it
  in full on every deepening attempt. §2.9's register-row rewrite targets exactly that.
- **A distance *bound* only seeds a prune limit, so a subsample is exactly as sound.**
  `mesh_to_mesh_distance` derived its bound by querying at every vertex — over 90 % of the call on a
  large mesh, to prune a traversal worth under 1 % of it — and a fixed-size stride sample gives
  bit-identical distances, because a looser bound prunes *less* and the narrow phase still sees the
  true argmin.
    - **An optimization that loosens a bound stresses everything that bound feeds**, and worked as a
      fuzzer here: it reached §12.2's `tile_bvh_query_aabb` overrun deterministically, and exposed a
      CUDA-only bug where the straggler re-walk *overwrote* the first pass's already-correct answer.
      **The CPU device, which runs no capped second pass, was the oracle that caught it** (§7.2).
    - A serial uncapped straggler pass was tried as the fix and reverted — §14.2's load imbalance is
      real and the block-cooperative walk earns its place.
- **Read the host half of a device profile before accepting a device-side attribution.** A
  `.numpy()[k]` on an array that scales with the mesh is the shape to grep for; its share *grows*
  with the mesh, the opposite of the usual falling-share decline.
- **DECLINED — the `cotmatrix` loss to pytorch3d is a scope mismatch.** `p3d_ops.cot_laplacian`
  returns an *uncoalesced* COO tensor with duplicates unsummed and no diagonal, where
  `laplacian.cotmatrix` assembles CSR with a row sum; coalesced to do the equivalent job, triwarp is
  ahead at every size. The *timing* is incomparable, not the result, so it stays a live parity
  comparison rather than a `noparity` exemption.
- **Apply the same detector to *both* sides' output before comparing costs.** The suite's largest
  reported loss for several rounds was a reference call that no-opped on that fixture, returning its
  input byte-for-byte while every collision remained — and a benchmark assert of "produced some
  output" cannot see it.
- **The hole-filling DP's span sweep is a chain of launches whose count is a floor — but they no
  longer have to be *issued*.** Invariant tables ride in the `HoleFillTables` bundle rather than
  being passed on a sweep of hundreds of launches, and the sweep itself is now **recorded once and
  replayed** (§14.3). At `rim_short`: **1.36x** on the benchmark row and **1.26x** on an isolated
  probe of the public call (§15.4 — the two are not comparable, so both are labelled), 1.5x on the
  sweep alone, flat at `holes_many`, and flat wherever the gate below declines. The launch *count*
  is unchanged — what changed is the price of each one.
    - **"Launch-bound" overstated the ceiling at the long-rim point, and that ceiling is what the
      capture spends.** On `rim_short` the 510-launch sweep read 6.06 ms wall against 5.13 ms of
      device time — overlapping, so removing *every* launch was worth ~16 %. It is now 1.5x on the
      sweep. The claim still holds where it was written — the many-rim end, where the sweep is one
      launch and `boundary`'s floor is everything.
    - **REFUTED, with the arithmetic — a blocked interval DP.** It was carried here for rounds as
      "unbuilt, would cut the launch count by an order of magnitude". It cannot pay, and the reason
      is occupancy rather than effort. Tiles on one tile-diagonal *are* mutually independent (cell
      `(i,j)` reads only `(i,k)` and `(k,j)`, both at a tile-diagonal ≤ its own), so the schedule
      is sound — but a diagonal holds only `B/tile` tiles against the `B - span` blocks a plain
      span level already launches, and that level width *is* what fills the device. Any scheme
      merging `C` levels bounds the block count at ~`B/C` and multiplies work by ~`C/2`. This is
      §14.9's persistent-block result from the other side, and the contrast with §14.11's stitch DP
      is the lesson: **tiling wins where the untiled schedule was *also* under-occupied, and loses
      where it was not.** The fill DP's levels are wide and cheap; the stitch DP's diagonals are
      narrow and trivially cheap.
    - **Per-span device time is nearly flat across a 127x work range** (7.1 µs at span 2 against
      10.4 at span 500, block 128; a null kernel at the same grid is 3.0). So "device time" here is
      mostly per-kernel fixed cost, not computation — which is why the sweep responds to the
      *number* of kernels and not to their shape, and why `block_dim` 32/64/128/256 sweeps to 128
      and stays there.
    - **REFUTED — transposed mirrors of the DP tables so the apex loop's second read coalesces.**
      Built, byte-identical, and a loss. The tables are L2-resident, so "one transaction per lane" is
      an L2 hit and the 32x transaction argument prices bandwidth this kernel is not paying; and the
      coalescing it buys was hidden under the sweep's launch cost where the two extra stores and
      two extra launch arguments were not. Recording the sweep removes the first half and leaves
      the second, so it does not reopen this. Do not re-propose without a rim whose tables exceed L2.
    - **REFUTED by a free PTX diff rather than a measurement — collapsing `triangle_fill_metric`'s
      duplicated geometry.** Its default branch recomputes ~40 % of its flops in *different* basic
      blocks, so removal needs partial-redundancy elimination rather than local CSE and there was no
      reason to assume it. **nvcc already does it**: hand-fusing changes three `sub.f32` out of ~1 100
      arithmetic instructions. **A PTX op-count diff costs no GPU time and no measurement.**

### 16.7 `sample`

- **Reach for randomized-priority selection whenever a GPU port needs a maximal-packing / MIS-shaped
  result.** `sample_surface_blue_noise` is randomized-priority parallel dart throwing, not Bridson:
  every pool point draws a priority, a point is accepted when no smaller-priority point still in play
  lies within `r`, and everything within `r` of an acceptance is discarded. The tie-break-free
  correctness argument — the later of any too-close pair was already discarded — is what makes it
  safe against a stochastic output, and it is the serial algorithm's own distribution.
- **The cover pass is load-bearing for *termination*, not just an optimization**: a wider shell or a
  looser cover radius can stop the loop converging, because the accept step will not take a point
  while a smaller-priority alive point still covers it. Its byte-identity gate is cheap insurance.
- **The round count is stable at 4-6 whatever the cloud** — over five mesh shapes and a 55x pool
  range, growing logarithmically as MIS theory predicts. "One cloud takes several times as many
  rounds as another" does not reproduce, and the loop around the two shell-scan kernels is near its
  launch floor.
- **A *second consecutive* readback is far cheaper than §13.1's queued figure**, which is the
  pipeline drain the first read already paid. §15.2's "price the candidate directly" applies to
  readbacks too.
- **Run the loop in cell-sorted index space.** The pool is already radix-sorted on its cell key to
  build the cell list, so that permutation *is* the cell order; permuting the payloads through it
  once at setup makes a cell's members the contiguous run its offsets already name, and every read
  stride 1. **The tie-break is what keeps it byte-identical**: the rule is "smaller priority, then
  smaller *pool* index", and renumbering changes what "pool index" means — so the comparison still
  reads the original index, on the priority-tie branch only.

### 16.8 `linalg`, `smoothing`, `laplacian`

- **REMOVED — batching several CG iterations per conditional-graph test.** The overshoot is a
  **fixed** number of launches whose *share* is set by how long the solve is, so long solves win a
  little and short ones lose a lot (best case +13 % on one cell, worst −40 % on three). The axis is
  the solve's own length, which nothing cheap knows in advance — which is why this is a removal and
  not a gate.
    - **The sweep that justified it could not express the failure mode.** Three of its six cells
      never reached the captured loop at all, their launch count identical at every value, so half
      the set was structurally unable to respond to the knob. **Read the launch count beside the
      ratio**: a cell whose launch count does not move under a knob is not evidence about it.
- **SHIPPED — `_BatchedCg` records its `capture_while` graph once per solver object and replays
  it.** It had re-recorded and re-instantiated the graph on every `solve`, which is the
  "record, replay once" row of §14.3 paid once per call. Legal because every buffer the graph
  touches is owned by the solver and never rebound, and `_initialize` resets the loop condition
  before each replay. **1.77x on `filter_laplacian(implicit_time_integration=True)` at ten passes
  and 1.15x on `arap`**, iteration counts identical. The guard is
  `test_spd_column_solver_reads_a_rewritten_rhs_on_every_call`: the older reuse test re-solved the
  *same* right-hand side, which a stale replay passes.
- **REMOVED — block conjugate gradient over two columns (`_BlockCg2`).** On well-conditioned systems
  a wash to +3 %; on ill-conditioned ones its iteration count goes *up* by 2.14x, because the two
  columns' search directions go nearly parallel and the shared subspace stops buying anything. A
  Tikhonov floor on the 2x2 Gram keeps it finite; only deflation/restart would recover the rate.
    - **A cheap perfect predictor existed and was still declined**: the Jacobi diagonal's spread
      separates the cases by three orders of magnitude, but a predictor is worth at most the +3 %
      best case, which does not pay for 309 lines and eight kernels (§4.2). **A mechanism whose best
      case is a wash does not need a better gate; it needs removing.**
    - **The lesson that outlives it**: it was validated on two well-conditioned saddles and shipped
      without ever running on `saddle_graded` — the fixture that exists *specifically* to be the
      ill-conditioned member of an otherwise identical pair, and exactly where block CG's documented
      failure mode lives. That is now §9's rule.
    - The regression guard is deterministic:
      `test_two_column_solve_costs_no_more_iterations_than_its_worst_column`, parametrized over a
      well- and an ill-conditioned shift. `_BatchedCg` satisfies it by construction; any mechanism
      that couples the columns does not. The old guard asserted the count was strictly *lower* and
      would have caught this had it been parametrized over the ill-conditioned arm.
- **A readback census parametrized by the loop count is the tell for a per-pass host sync**, and it
  is deterministic, so it settles the question on a busy box where a timing would not (§15.6). Patch
  `wp.array.numpy` and call at two iteration counts: a fixed base plus one per pass is the signature.
  An `ast` walk for a readback lexically inside a `range`/`while` loop reports ~25 live sites; most
  are genuine host *branches* (a convergence test, a compaction count) that cannot move to the device
  without changing when the loop stops. **Check whether the host branches on the value before
  proposing this** — `registration.icp_*`'s early exit is the worked counter-example.
- **When moving a host-side `if` into a kernel, check what the *skipped* branch used to do with the
  arguments it never evaluated.** An identity is only an identity for finite operands: on a
  face-less mesh the caller's centre is `NaN`, so a "skip" spelled as a scale of `1.0` propagates
  where the host version left the buffer alone. The whole suite was green across it — every fixture
  has faces. Sibling trap in the same edit: `wp.utils.array_sum` writes nothing for an empty input,
  so its `out=` accumulator must be `wp.zeros`, not `wp.empty` (§3.3).
    - The opposite case exists too: `filter_mut_dif_laplacian`'s correction writes
      *unconditionally*, because the host version it replaced applied an offset of exactly zero
      rather than skipping. **Answer the skip-versus-identity question from the code being replaced,
      not from the sibling.**
    - Its three volumes are summed by `wp.utils.array_sum` — deliberately the *same* reduction, not
      `measures.volume`'s tiled one — because the correction is a difference of two nearly equal
      volumes and mixing summation orders would surface there magnified.
- **`smoothing.inflate`'s inherited volume constraint costs ~2x and is load-bearing.** It restores
  the volume the *smoothing half-step* removed, so removing it changes what `pressure` means rather
  than only what it costs. Exposing it as a keyword was declined under §4.2: no in-repo caller wants
  it off.
- **DECLINED — capturing the multigrid MIS loop to drop its per-round readback.** The ceiling is ~2 %
  of the preconditioner build, against a medium rewrite with a documented reverted precedent
  (§12.2's `shortest_path_envelope`) and a device-infinite-loop hazard where the round cap is today a
  host `range`. The blocker does have a clean fix, recorded so it need not be re-derived: the odd
  `state` ping-pong is removable outright, because `mis_decide` reads only `state[i]` and can write
  in place.
    - **A probe that fakes a readback is faking every readback that reaches it.** Monkeypatching
      `read_scalar` across the build reported a 71 % saving while also faking `n_aggregates`,
      building a degenerate hierarchy that was cheap because it was wrong — and it never raised.
      Gate such a probe on a deterministic property of the result before reading its clock.
- **DECLINED — folding the `p.Ap` partials into `csr_matvec`.** That kernel is shared with the
  multigrid V-cycle at four sites that want no partials, and making it tiled would expose all of them
  to the ragged-tail hazard `cg_dot_partials` documents (a long tail cost several times a short one
  at nearly the same size, turning a 1.5x win into a 0.82x loss). Not worth one launch of six.
- **The multigrid hierarchy's *setup* is the blocker** — several sparse-op calls per coarsening level
  at Warp's fixed per-call cost, not the aggregation algorithm. Every losing case loses by exactly
  that; with a free setup all of them would win. **Cutting the per-call cost of the sparse
  matrix-multiply is the open lever.**
- **A numerical-method decision like "will the hierarchy pay for itself" belongs on a property of the
  *operator*, not of the problem size or an extrapolated iteration count.** Both of the latter were
  tried first because they are cheaper to compute and both were wrong; off-diagonal dominance
  separates the cases.
    - **Never route a new caller through the `"auto"` gate without re-measuring on its own systems.**
      An operator with a favourable dominance reading on one mesh can read unfavourably on a
      differently-shaped mesh of identical connectivity. The open lead is a strength-of-connection
      threshold for anisotropic operators; the connection-Laplacian operator cannot reach the gate at
      all, because the hierarchy is scalar-CSR-only.
- **Every direct-factorization reference is flat across the conditioning axis and triwarp's CG is
  not**, so a conditioning regression in triwarp is an **iteration-count** problem, not an assembly
  or operator problem. Three independent implementations agreeing on flatness is the strongest
  version of this the suite has. `isotropic_remesh` **inverts** it — triwarp flat, the reference
  tripling — because triwarp runs a fixed `iterations` x five launches whatever the input looks like,
  so **its flatness is a fixed work budget, not insensitivity** (§16.4).
- **A direct GPU factorization (cuDSS) wins only where the multigrid gate already fires**, and is not
  installed. **The cost is the symbolic plan, not the numeric work** — flat in conditioning, so it
  wins exactly where `CG_MULTIGRID_DOMINANCE` already routes to multigrid and loses badly everywhere
  CG converges quickly. **Do not add it as a one-shot backend.** The real lever, unmeasured, is plan
  reuse for ARAP-shaped loops — and `parametrization.arap` already beats igl in all six benchmarked
  cells, so no loss row justifies the dependency. Traps if it is ever opened: AMD reordering beats
  the defaults without the MT layer; a one-shot `direct_solver` re-creates the handle;
  `reset_operands(a=new)` drops the plan; `libcudss.so` is not on the loader path; and systems with
  **empty rows** (unreferenced free vertices) are singular for a direct solver where CG leaves them
  at the initial guess. A host launch count cannot bound the solve's share, because the CG is
  graph-captured (§15.10).
- **CLOSED — `robust_laplacian`'s negative-weight residue is entirely *boundary* edges**, which are
  not Delaunay violations at all: a boundary edge has one opposite angle, so the two-angle condition
  has nothing to compare against. The only remedy is Steiner points, which the contract forbids. The
  interior half is fixed by `intrinsic_delaunay`'s multi-edge support — a flip may create a second,
  geometrically distinct edge between already-adjacent vertices, which intrinsically is a different
  geodesic and not a duplicate.
    - **Quoting a `min()` over a set whose members have two different causes can size a proposed fix
      by 100x the wrong number** — the residue's largest member was the unfixable boundary case.
- **A `for` loop around a single-column solver, in a module whose siblings call the batched one, is
  the textual tell for a multi-column solve.** All three position components share one operator, so a
  batched solve advances them in one Krylov iteration whose count is set by the worst column rather
  than the sum of three.
- **A helper that returns the same sentinel for two logically different "nothing to do" cases is a
  defect waiting for the rarer case.** `filter_implicit_fairing`'s Dirichlet path returned "no
  constraint" both for "nothing pinned" and for "no free interior vertex", so pinning the boundary of
  a single triangle or a small hole patch silently ran the unconstrained solve.
- **`smoothing`'s conditional-emit triplet writers must point unwritten slots out of range, not to
  zero** — §12.7, an order-of-magnitude cost fix reached by `holes.fill_smooth` through a shared
  helper.

### 16.9 The 0.3-5.0 ms loss band, attributed

**Attributed per row before touching any of it, and the band is mostly closed or floor** — the
largest genuinely-open rows were addressed by other items. That is the argument for attributing
first and expecting the row to belong to someone else's item. Per-segment floors (`split_array`,
`concatenate_arrays`, `pack_1d_arrays`) are §13.1's constants; `is_watertight` is a documented scope
mismatch; `marching_triangles` was a median artifact at `rounds=3` (§15.4); `cluster_decimate`,
`remove_degree3_vertices` and `lscm` are not work items. **The band is computed from medians — run
`aggregate.py --suspect` first.**

**Three of the four rows that had no attribution are *device*-bound, the opposite of what this
section's method reported**, because they graph-capture and `wp.timing_begin` cannot see that
(§15.10). So the remainder is not launch overhead and launch elimination will not touch it.
`transport_tangent_vectors` / `vector_heat_scale` are 71 % device, of which two `TiledDot` kernels
are 42 % of the solve — §12.7's batched-dot finding on the *unbatched* path.
`delaunay_triangulation` is the one genuinely host row, and its host half is the *designed*
single-thread CPU seed.

- **Before optimizing a traversal, read what its caller unpacks.** `homology_generators`' BFS ran
  four kernels per level; the one in-tree consumer unpacked `order` and used it for its `.shape[0]`
  alone, while two of those four kernels existed only to compact claims into scipy's FIFO order.
  Replacing the tie-break with a `wp.atomic_min` on the parent keeps the tree deterministic, halves
  the level, and — because the tie-break no longer reads column position — removes the requirement
  that the adjacency be *column-sorted*, which is what let the CSR build drop `bsr_from_triplets`
  for a degree count, a scan and a cursor scatter. The depth array the loop wrote anyway sizes each
  generator's loop without walking it, so the Python tracing became two launches.
    - The basis *itself* differs: the atomic-min tree is not the FIFO tree, so total loop length
      rises ~10 % at genus 64 for the same generator count. A homology basis is not unique and the
      suite's invariants cover it, but a caller feeding `shorten_loop` pays that.
    - **`graph.bfs`, `bfs_from_edges` and `bfs_multi_source` are deleted** — none had an in-tree
      consumer once the homology decomposition stopped calling one, and reproducing
      `scipy.sparse.csgraph.breadth_first_order` exactly is a promise nothing needed.
      `kernels/algorithms/bfs.py` keeps only `per_source_bfs_collect`, the geodesic-ball engine.
      Their benchmark groups went with them, retiring `bfs[ribbon_long]` — which §14.9 had measured
      as ~95 % one-thread serial drain and unreachable from any triwarp call. **Deleting the entry
      point is the honest close, not a win.**
- **A published attribution can outlive its own fix.** `combine.split`'s prescription — "the
  per-component allocation/launch sequence needs batching" — is stale: that batching shipped as
  `split_batched`, and `split` now issues exactly the same launch count. What remains is §13.1's
  per-slice-view cost over thousands of returned arrays, and building those view objects *is* the
  return value. **Before batching a "per-component wrapper chain", count its launches against the
  batched sibling.**

### 16.10 Where triwarp wins big (for context when reading a loss)

`screened_poisson` 15-25x over open3d, `combine.split` on few-component meshes 34-120x,
`winding_number` 13-18x over igl, `procrustes` ~7x over trimesh (but only even with open3d — it is
launch-latency bound, not compute bound), `cluster_decimate` 9x at `dragon`, `polyline_simplify`'s
level-synchronous RDP two orders of magnitude at `rim_long`. Small-input fixed overhead of device
reductions and scan pipelines costs tens of microseconds against the host path — accepted, because
large meshes win 10-260x.

**A third independent implementation is what makes an outlier legible**; trimesh alone is slow
enough everywhere that a 3x triwarp regression still looks like a win against it. That is the
argument for carrying nine references.

### 16.11 The homology de-duplication pass, and what a shared convention costs

A fusion round leaves kernels that are a second spelling of something the package already had. What
was reused rather than kept, as a checklist for the next such pass: a `dim=1` seed write was
`scatter.scatter_index`; a per-vertex degree count was `scatter.count_occurrences` over the
flattened rows; two bespoke loop-state slot tables became the shared `LOOP_ROUND` / `LOOP_CONDITION`
with the extra slot **appended**, and three `dim=1` advance kernels became one `array.loop_advance`;
a private unsorted-CSR builder became `graph.edges_to_neighbor_lists`; and a `_run_device_loop`
copied verbatim at three sites became `_device.run_device_loop` — whose fallback two of the copies
had wrong (a hand-rolled `while` testing *after* the body, where `wp.capture_while` tests before).

**Such a pass is gateable byte-for-byte** precisely when the algorithm's tie-breaks are by index, so
they cannot see a row permutation. **It also costs a little**, and the cost is attributed rather than
residual: a shared advance kernel is fractionally dearer per round than the three-statement one it
replaces, and reordering a count ahead of a guard readback loses the overlap with the host's graph
recording. Accept it deliberately, with launches, allocations and readbacks held equal.

- **A whole-call A/B cannot resolve a region worth a fraction of a percent of the call**, and read
  the sign backwards twice here (§15.2's pessimistic direction). The verdicts that held came from
  deterministic counts and from timing the changed kernels alone. A cProfile `allocate` delta was an
  artifact of aggregating two same-named functions; instrumenting the allocator directly gave the
  same count in both arms.
- **A row in the A/B that the change does not touch is the control that licenses the rest.**

### 16.12 `holes`

- **A packed engine fed by the split form pays the loop count twice, and nothing in the signature
  says so.** `triwarp.holes` builds every stage on `_PackedLoops`, but two of the three places that
  derive rims from a mesh took `boundary_loops` — which is `split` over
  `boundary_loops_batched`'s buffer — and then concatenated the per-loop views straight back.
  At ~3 µs to build a view and ~6 µs to copy a segment (§13.1) that round trip is linear in the
  rim count on both sides: measured **37.3 ms against `_hole_loops`' 2.15 ms** on an 8 192-rim
  sphere, for the identical answer. Both now go through one `_boundary_loops_packed` helper, worth
  **2.0-2.1x at 512 rims and 15-16x at 8 192** on `extend_hole` / `build_bottom` and nothing at 2
  rims. **The tell is a wrapper that calls the split form and packs the result** — grep for
  `boundary_loops(` next to `pack_1d_arrays` / `concatenate`. The third, `_single_boundary_loops`,
  keeps the split deliberately: the whole `stitch*` family requires *exactly one* rim per mesh, so
  a many-rim input raises rather than paying, and §4.2 says an axis with no caller is not built.
    - **The benchmark could not see it**, because `test_extend_hole` and `test_fillable_loop_mask`
      hoist `boundary_loops` out of the timed callable and pass `loops` — correctly, since the
      derivation is not the op. A hoist that removes a *cost* also removes an *axis*; say so in the
      row's docstring when the default path does not take the hoisted route.
- **`fillable_loop_mask`'s two pinch tests are one per-vertex occurrence count**, not a
  distinct-vertex test per loop plus an owner tally over their unions. A vertex holding two packed
  loop slots is visited twice by one rim or once by each of two, and every rim touching it is
  unfillable either way — so `count > 1` is exactly the union of the two conditions. That is
  §2.4's duplicated *decision rule* one level up: two host passes, each with its own `np.unique`,
  expressing one rule. The whole predicate is now readback-free (it returned a `wp.array[wp.bool]`
  already and was building it from NumPy). **The predicate itself measures 25.7x on two
  65 536-vertex rims** (22.3 ms -> 0.87) and **3.9x at 512 rims**; end to end, where
  `boundary_loops_batched`'s flat ~2.1 ms floor is the rest of the call, that reads **9.0x**,
  **3.2x**, **32.4x** at 8 192 rims and **2.2x** on a 407-rim scan mesh. `np.unique` alone was
  85 % of the long-rim call — 21.4 ms of 25.1 — which is §3.8's "NumPy reducing a full
  readback" wearing a set operation's clothes.
    - The range guard is fused into the counting kernel rather than left to the caller, because the
      histogram is indexed by the value read (§12.1). An out-of-range rim now answers `False` where
      the host form raised an undocumented `IndexError` from NumPy's own indexing.
- **The min-weight DP's traceback runs on the device, one thread per rim.** The host form read the
  whole `sum(B^2)` predecessor table back to reach `sum(B)` of its entries and then built the
  triangles a Python tuple at a time: **0.765 ms at two 512-rims and 0.804 ms at 512 three-rims**,
  the latter 22 % of the whole call. The device walk is **2.9x / 5.8x** on those, taking
  `fill_min_weight` 1.24x and `fill_small` 1.26x at the many-rim point. It is byte-identical — the
  DFS takes `(k, j)` before `(i, k)` exactly as the Python stack did — and that is checked on 611
  CPU outputs, not inferred.
    - **A rim can emit fewer than `B - 2` triangles** (a forbidden chord, or a pinch, leaves
      `prev = -1`), so each rim writes into its own padded block and reports a count. The single
      readback is the packed total, which sizes the return *and* tells the caller whether any rim
      fell short — so the compaction launch only runs when one did.
    - Its stack is caller-allocated scratch, one slot per packed rim vertex: the walk starts one
      deep and each emitted triangle nets one entry, so the depth never passes `B - 1`.
- **`fill_min_weight`'s triangulation is not bit-reproducible on a degenerate rim, and never was.**
  `loop_rim_metrics` accumulates each rim's Newell normal with float32 `wp.atomic_add`, so the
  normal varies in its last bits run to run (measured: 2 distinct results over 30 launches on one
  input), and on a rim whose metric has exact ties — a pinched non-manifold rim, or two rims of
  equal perimeter under `preserve_largest_hole` — that flips which triangulation wins.
  `_PackedLoops.perimeters` has the same shape and gave 15 distinct values over 30 calls on a
  cylinder whose two rims are equal. **Consequence for any A/B on this module: gate on the CPU
  device**, where the atomics serialize and 611 outputs compare byte-for-byte, and do not read a
  CUDA difference on a degenerate fixture as a regression.
- **SHIPPED — the stitch band DP is a blocked wavefront, and a launch traded for a block barrier
  is worth ~40x here.** It was the module's one launch-bound row: `n_a + n_b` anti-diagonals, one
  launch each, **2 049 launches and 24.6 ms of a 29 ms call against 7.0 ms of device time** at two
  1 024-vertex rims. `StitchTables` had already collapsed the per-launch argument cost and graph
  capture is refuted for a sequence issued once (§14.3), so the launches themselves had to go.
  `stitch_dp_tile` gives one **block** a `32 x 32` square of the grid and launches one
  tile-diagonal at a time, so all but `2 * ceil(n / 32)` of the sequential steps become block-local
  barriers. Measured on the DP sweep alone: **11.0-13.3x on CUDA** at rims of 128 to 2 048 and
  **2.4-7.8x on CPU**. End to end and byte-identical: `stitch_loops_min_weight` **1.76x at 64-rims
  rising to 6.94x at 1 024**, `stitch_min_weight` 1.29-4.14x behind its `boundary_loops` floor,
  CPU 1.18-2.35x; the benchmark row, which is a harness number and not comparable with those
  (§15.4), reads **2.9x** and turns a 1.2x loss to meshlib into a 3.6x win. The host side was never
  the lever — `came.numpy()` is 0.27 ms for a 4.2 MB table and the band traceback is sequential.
    - **`wp.tile_sum(wp.tile(x))[0]` is the barrier, because Warp exposes none** (§10, §13.2). It
      orders global writes across warps, which is not a thing to assume: **probed by removing it**,
      and a 2-D wavefront then fails 8/8 runs at 64, 128 and 256 lanes and passes 8/8 at 32.
    - **32 lanes is the optimum and it is also the width at which the barrier cannot be tested.**
      A 32-lane block is one warp, so it needs no barrier at all — which is exactly why it is
      fastest (`tile_reduce_impl`'s `warp_count == 1` fast path, 126 ns against 325 at 64, §12.6)
      and exactly why `test_stitch_dp_tile_matches_diagonal` parametrizes `tile` over 32 **and**
      64. Deleting the barrier leaves the 32 arm green and fails the 64 arm on every run. **A
      tuning constant that makes a correctness mechanism unobservable needs the test to cover a
      value it does not ship.**
    - **The tiled schedule wins on the CPU device too, so it is the default on both** — unlike the
      fill DP's tiled engine, which CPU declines. There the lanes are the win and CPU has one; here
      the win is the launch count, which CPU pays for as well. Its own optimum is 64 rather than 32
      (4 % apart), which §13.3 calls inside the tolerance, so the constant is deliberately not
      device-split.
    - **There is no crossover to gate on**: the tiled form is ahead at every rim from 10 vertices
      (7.1x CUDA, 11.6x CPU) to 2 048. `stitch_dp_diag` stays as the reference schedule the test
      compares against, reached through `_run_stitch_dp(tiled=False)`.
    - **A `@wp.struct` type cannot be annotated**, because the decorator rebinds the name to a
      `Struct` *value* — so a helper taking a bundle names `warp._src.codegen.StructInstance`
      instead, and basedpyright's `reportGeneralTypeIssues` is what says so.
- **Every other NumPy site in the module was priced and kept.** The `stitch_loops` monotonicity
  correction (a longest-increasing-subsequence over the association array), the band traceback and
  `bridge_edges_smooth`'s Hermite strip are host-*sequential* or fixed-size, which is §3.8's
  sanctioned bucket; `_PackedLoops`' cumsums and uploads are 0.10-0.16 ms even at 8 192 rims.
- **The floor under every entry point here is `boundary_loops_batched`, at a flat 1.9-2.1 ms**
  whatever the mesh — which is `triwarp.boundary`'s own floor and is recorded as closed (§16.5).
  `fill_fan[holes_many]` is 2.2 ms of which 2.1 is that call, so a filler row at the small end is
  measuring `boundary`, not `holes`.

### 16.13 The whole-`kernels/` sweep (2026-09-22)

Eight reviewers, one per disjoint group of kernel modules and their wrappers, working in the live
tree at once. What made that workable, and worth reusing: **file ownership was disjoint** (a
change needing another group's file was reported, not made); **evidence was counts only**
(launches, allocations, copies, readbacks, from a monkeypatch census at two sizes), because eight
processes sharing the GPU make any clock a fiction (§15.6); and **each byte-identity gate ran on
an overlay** — the baseline worktree with only that reviewer's files laid over it — because the
live tree's counts mixed every reviewer's savings. The clock came last, on a quiet box, one A/B
over 69 public calls in alternating processes, reading the `min` of three runs per arm.

**Result: no call regressed, and the ratio tracks how host-bound the call was, exactly as §16.1
predicts.** 2-3.8x where a whole pipeline collapsed (`connected_component_labels_from_edges`
3.09x, `vertex_normals` 2.84x, `polyline_angles(closed=True)` 3.77x,
`remap_discrete_attribute_from_uv` 2.59x, `uv_seam_vertex_mask` 2.21x,
`face_defective_mask` 2.17x); 1.2-1.6x across the mid-level surface (`split_batched`,
`subdivide_to_size`, `split_edges`, `face_adjacency_convex`, `boundary_loops_batched`,
`fill_min_weight`, `cluster_decimate`, `filter_laplacian`); and **flat (1.00-1.02x) on every
device-bound call** — chamfer, Hausdorff, `max_tangent_sphere`, `normals_at_closest_faces`,
`mesh_to_mesh_distance` — even where their launch counts fell by a third. That is not a failed
optimisation; it is §15.2's share rule. `isotropic_remesh` 1.17x and `fix_self_intersections`
1.12x are the large-call cases where the removed work was a real share.

Findings that generalise past this pass:

- **An edge-parallel union-find replaces a CSR build whenever the answer is "smallest id per
  component".** `ecl_hook_edges` hooks the larger root under the smaller, so every label is the
  component's minimum node id whatever order the unions ran in, and the labels equal the CSR
  path's exactly. It drops `bsr_from_triplets` and its radix sort from every caller
  (`face_connected_component_labels` 1.65x, `boundary_loops_batched`, `vertex_manifold_mask`,
  `successor_cycles`).
  **Shipped without its pre-hook, it regressed the two fixtures built to break it (§16.14).**
  From a bare identity forest the CAS hooks race: `rim_long`'s sequential-id cycles grow parent
  chains as long as the cycle and `fan_hub`'s 40 960 spokes retry against one root, taking
  `boundary_loops[rim_long]` 3.6 -> 26.4 ms and `is_vertex_manifold[fan_hub]` 2.0 -> 9.1. One
  `wp.atomic_min(parents, max(a, b), min(a, b))` per edge first (`ecl_init_parent_edges`) changes
  no root and fixes both: **1.6 / 0.73 ms**, past the CSR path's own figures, for one extra launch
  (~0.02-0.04 ms) on an ordinary mesh. "Changes no root" was true of the answer and false of the
  cost -- §9's fixture-pair rule, a fourth time.
- **`array_cast` is generic, so `astype` pays generic dispatch on every call.** A census of
  `array_cast` over the suite put four dtype pairs at 97.9 % of calls; `kernel_array.ASTYPE`
  launches concrete kernels for those and falls back for the rest (1.56x on the call).
- **A derivation already paid for is often recomputed one call later.** `metrics` reduced the
  same array for the forward maximum and again for `"max"` Chamfer; `triangulate_point_cloud`
  sorted and hashed the same rows twice; `isotropic_remesh` built an edge grouping its next stage
  rebuilt; `voxel_down_sample` probed point slots pooling had just probed. Grep a wrapper for two
  calls fed the same arguments before looking for anything cleverer.
- **Three call-site conversions the shared-library changes enabled**: `mask_to_compact_ranks`
  now scans in place, so `counts_to_offsets(astype(mask, wp.int32))` is strictly worse and was
  converted at every site; `unique_1d` returns its values as a prefix view of the sort scratch,
  so a returned array keeps a buffer twice its length alive — a deliberate trade of memory for a
  copy; and `read_scalar` copies with `src_offset=` from a contiguous rank-1 source, which is safe
  because the destination is still pageable (§12.1's race needs a pinned one).
- **Declined as noise, with the reason at each site**: fusing the multigrid V-cycle's matvec into
  its Jacobi sweep (inside a captured loop, §14.10), graph-capturing ball pivoting's 8-wave batch
  (needs a device/wall split first — likely device-bound), and every ICP-loop fusion beyond the
  per-iteration memsets (the loop reads a scalar back each iteration, so the clock stays flat).
- **The three fusions that needed an allowlist entry rather than more code, resolved.** A kernel
  that clears state it has just read is the same launch as one that does not, so what blocks it is
  check 13's `out_` rule, and the entry is the whole cost. Two shipped, byte-identical on CPU:
  `sample.apply_deletions` now clears `alive` in the pass that subtracts the deleted points'
  contributions (one `wp.map` per elimination round; **1.27-1.30x** on
  `sample_surface_poisson_disk`), and `homology.forest_link` clears `candidate[e]` for an accepted
  edge instead of writing an `in_forest` mask the wrapper then subtracted (one allocation and one
  map; **1.03-1.04x**). Both are races only on paper: each thread writes its own slot, and every
  reader of a written slot is already excluded by a second test. **`arap`'s rotation right-hand
  sides stay zeroed by the wrapper** -- the two memsets are ~0.15 % of an iteration the CG solve
  dominates, which does not pay for turning two read-only inputs into in-place state (written at
  `kernels/parametrization.arap_interior_rhs`).

### 16.14 Benchmark round 14 (2026-09-23)

Items from `plans/benchmark-round-11.md` (round 14), each measured back to back against a detached
baseline carrying every *other* item, with byte-identity gated on the CPU oracle (§7.2). CUDA
differences that also appear baseline-against-baseline -- `quadric_decimate`, `ball_pivoting` on
`bunny`, `remove_degree3_vertices`' replacement-face order -- are the pre-existing float-atomic and
atomic-cursor order, not the change.

- **`remove_tunnels` cut a separating family (correctness).** Vertex-disjoint non-trivial loops can
  still be *dependent*: on `handles_64` 23 disjoint shortened generators bound a piece of the slab,
  and cutting all 23 split it in two (chi -126 -> -78 where 23 cuts claim -80). The basis changed
  in `2d848b9` and the selection had never checked independence -- "a basis is not unique and the
  invariants cover it" was true of the basis and false of the subfamily. Fix: label the faces of the
  cut mesh; every component after the first is a dependent loop, and a longest-first spanning
  forest over the component graph (components as nodes, loops as edges) leaves exactly those uncut.
  22 removed, invariant holds. **None of `_handles`' smaller variants (3-10 holes a side, three edge
  lengths) produces a dependent family**, so the unit test carries the genus-64 slab itself.
- **The largest cluster was a preconditioner, not launches.** `smooth_region` solves normal
  equations `MᵀM` -- squared, fourth order -- and the free rows of `M` are `M_ff = D⁻¹ L`. The
  *exact* `M_ff⁻¹ M_ff⁻ᵀ` converges in 21-30 iterations on every system probed (against Jacobi's
  421-5 000 and the `MᵀM` V-cycle's 153-313), so the one ring of rim rows `M` adds does not spoil
  the spectral equivalence. Approximating `M_ff⁻¹` is the whole design question, and three were
  built:
    - **A V-cycle on `L_ff`**: 77 / 126 iterations, but its hierarchy setup is **13-14 ms**, half of
      `bunny_decimated`'s call before one iteration -- §16.8's setup-is-the-blocker again.
    - **A Neumann series**: needs no setup and is weak, because `D⁻¹L`'s spectrum reaches 2, where
      `(1 - λ)ᵏ` does not decay.
    - **A Chebyshev polynomial on `[a, b]`, shipped** (`linalg.squared_laplacian_preconditioner`,
      *with a recurrence bug found in §16.15 that its `theta ~ 1` hid*,
      degree 12, `a = 40 / n`): one launch per step in the three-term `x` recurrence, no setup.
      `smooth_region[bunny]` 69.5 -> 17.0 ms (4.1x), `[bunny_decimated]` 2.4x, `saddle` /
      `saddle_graded` regions 4.1-4.5x, `fill_smooth[rim_short]` 1.52x; positions within float32
      rounding. The best lower end falls as `1/n` (0.04 at 1 000 unknowns, 0.005 at 9 000).
  **The upper end must be the Gershgorin bound, not 2.** Clamped cotangent weights go negative on a
  regular grid's near-right triangles, which puts `M_ff`'s spectrum at 2.18, and a Chebyshev
  polynomial fitted to `[a, 2]` explodes past its interval: the uniform `saddle` took **2 824**
  iterations (3x slower than before) while its graded twin improved. With `b = 1 + max_i Σ|L_ij| /
  L_ii` (the existing `offdiagonal_dominance_rows`) it is 153. §9's pair rule caught it; the unit
  test's negative-weight arm reads 340 against 37 under the mutation.
- **A `k = 1` query far from its cloud is a closest-point query.** Warp exposes no node-by-node
  traversal (§12.2), but it does not need to: a `wp.Mesh` whose triangles each collapse onto one
  point (`(i, i, i)`) answers `mesh_query_point_no_sign` with the nearest *point*, exactly -- the
  closest point of a collapsed triangle is its corner, and distances matched the grid's bit for
  bit. It is the BVH's pruned descent, so its cost does not grow with the displacement: 187 ms ->
  4.2 on `dragon` shifted 5 % of its diagonal. **It loses on queries on the cloud** (0.74 -> 1.76
  ms on `dragon`, the tree build), so it ships as a *deferral* inside the grid kernel at `k = 1`:
  a row that would take the linear scan is marked `DEFERRED_ROW` and counted, and one 4-byte read
  decides whether to build the tree for them. `chamfer_points_to_points[dragon oneway]` 186 -> 6.9
  ms (27x), `[bunny oneway]` 2.9x -- the 1.51x loss to pytorch3d-cuda becomes a win. Three facts
  that decided the shape:
    - `bvh_constructor="lbvh"` is the only viable one: `sah` / `median` build it 30x slower.
    - **Bringing the deferral radius in is a trade, not a win**: at 0.25-0.5 of the scan cutover it
      halves displaced rows' walk, and it defers the odd outlier of an *on-surface* query set,
      whose one row then pays for the whole tree (`bunny` on-surface 0.35 -> 0.50 ms). Kept at the
      cutover.
    - **§16.6's backward seed is now capped at the target's density estimate.** Seeding the
      backward search at the forward half's largest distance saved deepening rounds when the
      pair is a fraction of a spacing apart, and at 5 % displacement it made cells so wide that
      certified rows scanned thousands of points (59.9 ms against the default's 7.6). `min(seed,
      knn_initial_radius)` is best of both at every cell swept: `chamfer[dragon symm]` 245 ->
      15.8 ms, `hausdorff[dragon]` 239 -> 15.6.
- **`ball_pivoting`'s seed kernel re-proved the same failures every wave.** 21 of its 28.6 ms were
  waves 90-133 at a flat 0.54 ms each: already-failed orphans re-walking and re-testing every pair.
  An attempt's outcome depends only on the stored neighbour list (`ball_is_empty` never reads
  `point_used`), so a point whose failed walk saw its *whole* neighbourhood can never succeed and
  returns at once (`SEED_EXHAUSTED`); a truncated walk is not remembered, since used neighbours let
  the next walk past the break. The deferral cache the plan proposed first was sound and bought
  nothing -- few threads take that return. `seed_triangles` 28.6 -> 12.7 ms, `[bunny_decimated]`
  1.49x, byte-identical on CPU.
- **Flip rounds replay as a plain graph** (see §14.3): 13.1 -> 8.9 ms on a 50-round call; the
  interior-edge count cannot change after round 0 (every candidate kernel opens with the
  duplicate-edge guard), so later regroups skip its readback. `quadric_decimate`'s two readbacks
  per pass became one (a fourth `state` slot): 1.02-1.03x, as expected for a read queued behind
  work the first already drained. **Revised in §16.15**: round 0 is now issued and the graph
  recorded from round 1, because a call whose first round flips nothing paid a whole recording.
- **Region-sized versions of region questions.** A kept rim edge was an input boundary edge
  exactly when no *deleted* face contains it (its input count is `1 + deleted faces containing
  it`), so `delete_region_keep_boundary` could sort `3k` region keys instead of grouping every mesh
  edge: 1.09-1.12x, identical on simple rims -- and **wrong on a pinched deletion, so not taken**.
  A pinched rim is not a successor graph (the pinch vertex has two outgoing boundary edges), and
  `boundary_loops_batched` hands it to `graph.successor_cycles` anyway, whose Notes say colliding
  ranks leave slots at `0`: the loop comes back with `(0, 0)` edges, on both devices, at HEAD. Those
  fake edges match no deleted face, so the inverted rule reads the new rim as the input's and drops
  it (the CPU pass caught it); the existing rule fails safe by reporting it. **The unit test's
  "interior" region was itself pinched** -- the first six faces off the rim are not contiguous, 14
  hole edges over 10 vertices -- so it had been passing on a garbage loop.
  **Fixed since:** `boundary_loops_batched` now walks a pinched rim as a successor graph over
  boundary *halfedges* (`kernels/halfedge.next_boundary_halfedge`, moved there from `repair`),
  gated on the degree flag it already read, so the common path is unchanged. The loops are the
  boundary of the surface with each pinch vertex split once per fan: every boundary edge exactly
  once, in winding, and a loop can pass a pinch vertex twice where two holes touch. The unit test
  grows its regions over shared edges and asserts they are simple, and the pinched deletion is its
  own test. That removed the blocker, and **the deleted-face rule is now in** (1.15-1.17x,
  byte-identical). A runtime guard -- count each loop edge's kept faces with the `_EdgeTable`
  probe, and report any loop with an edge not held exactly once as new -- was built, bit, and
  measured **flat** (0.97-1.04x): the probe and its sort cost what the mesh-sized table did. It is
  also provably redundant, so the guarantee lives in a test instead. Twins are true opposites or
  `-1` even unvalidated, so every sector-walk successor starts where its predecessor ends; and a
  rotation enters a face only through the twin of its incoming halfedge, which a boundary
  halfedge lacks, so no two rotations merge and `successor_cycles` never gets colliding input.
  Every consecutive pair any walk returns is therefore a boundary edge on *any* mesh. What a mesh
  that is not edge-manifold can do is **lose** a loop -- a rotation meeting a three-faced edge
  dead-ends -- and a missing loop defeats every classifier equally.
  `test_boundary_loops_never_invent_an_edge` pins that (three fin orientations at a pinch, both
  devices; all six fail under the old vertex walk). **General lesson: when a runtime check costs
  the win it protects and a proof covers it, move the check to a test.** **Kept:** the
  empty-deletion early exit, read off the kept face count with no readback -- 2.51 -> 0.42 ms, since
  it skips the loop extraction too. **`holes._EdgeTable`'s inversion is in**: sort the rim's own
  keys and let one face pass probe them, in place of a radix sort of every mesh edge and two
  clones. Exact, and within noise on the hole chains (interleaved, never slower) -- the removed
  work is mesh-sized traffic on a launch-bound call, so it ships for the work it removes.
- **Small items.** `refine_and_smooth_region` derives the free ranks and unique edges once for its
  two solves (~0.5-1 ms of a hole-chain call). `remove_degree3_vertices` validates topology on pass
  0 only (`halfedge_twins` / `vertex_one_rings` gained `validate=`) and knows its kept count
  (`n_faces - 3 n_selected`): 1.18x on `dragon`. `face_self_intersecting_mask` runs its narrow
  phase inside the marking kernel (one launch and one per-pair buffer fewer; within noise on a
  BVH-bound call, byte-identical).
- **Three items first recorded as "not done", re-evaluated and done.** The recorded reasons did not
  hold: the precision objection ignored that `tw.reduce.sum` already commits its tile partials
  with `wp.atomic_add` (order-dependent on CUDA), "CG-bound rows" is §15.2's share argument about
  what a benchmark can *see* rather than whether work is removed, and the spectral-radius item was
  never tied to the `L_ff` hierarchy -- the smoothed-aggregation hierarchy still serves `harmonic`
  at `k >= 2`, `"auto"` and `"multigrid"`.
    - **`heat_operators`' timestep** reads the mean unique-edge length off the Laplacian's strict
      upper triangle (one entry per edge, twelve triplets per face, nothing pruned) instead of
      re-sorting every edge with `edges_unique`: `heat_operators` **1.06-1.21x**,
      `vector_heat_operators` 1.06-1.07x, `heat_geodesic` 1.02-1.03x. `t` moves by ~1e-7 relative
      (float64 accumulation instead of float32), and a zero-length self-edge of a degenerate face
      no longer enters the mean. **The scalar and vector solvers read the same kernel off two
      operators with one sparsity, so their `t` is bit-identical** -- the agreement `log_map`
      depends on holds by construction rather than by calling one helper twice. A test that
      pinned `t` to `mean_unique_edge_length` at 1e-12 now recovers `t` from the system and
      checks the convention against trimesh's unique edges at 1e-6.
    - **`reconstruction._mean_positive_finite`** folds sum and count in one pass into one
      `float64` pair: one launch and one readback where it took two maps, two reductions and two
      readbacks. The spacing moves ~1e-7 relative; `ball_pivoting`'s auto radius was already
      documented nondeterministic (§16.3).
    - **The multigrid damping** is folded into the inverse diagonal on the device
      (`kernels/algorithms/multigrid.damped_inverse_diagonal`), so no level reads its spectral
      radius back. Every consumer's arithmetic is unchanged -- `(omega * inv) * r` and `(-1) *
      (omega * inv)` are the products it formed -- and the hierarchy is **byte-identical on CPU**,
      where the device `pow` is the host's C `pow`; on CUDA the device `pow` moves `omega` in its
      last bit (a 1.3e-12 change in one level's cycle output). Flat on the clock (0.99-1.01x): the
      removed sync was queued behind the aggregation's own count readback.

### 16.15 Benchmark round 15 (2026-09-24)

Items from `plans/benchmark-round-11.md` (round 15), each timed against a detached `HEAD` worktree
through `plans/baseshim.py`, byte-identity gated on the CPU oracle where the schedule was the
change. Probes are in `plans/benchmark-round-15-data/probes/`.

- **A single-level Jacobi-Chebyshev preconditioner wins on long Laplacian solves** --
  `linalg.chebyshev_preconditioner` and `preconditioner="chebyshev"`, `z = p(D⁻¹A) D⁻¹ r` at degree
  12, one fused launch per step. This refutes the "Chebyshev as a single-level preconditioner is
  also worse" line `linalg`'s module docstring had carried: the `sqrt(k)` argument behind it counts
  mat-vecs, and a solve here is bound by launches. Against `HEAD` (harness, min of 2): `lscm`
  2.8-3.4x, `harmonic` k=1 1.4-1.5x and `harmonic_conditioning` 1.9x / 3.0x, `heat_geodesic`
  1.4-1.9x on the medium/large spheres and 2.1x / 3.3x on the conditioning pair, the
  `heat_signed_distance*` family 1.4-3.1x, `log_map` 1.8x / 2.8x, `filter_implicit_fairing`
  1.5x / 2.5x, `arap` 1.8-2.7x on the saddles. Measured losses: `arap[hemisphere 10]` 0.83x and
  `heat_geodesic[sphere_small]` 0.96x.
    - **It is opt-in per call site, and must stay so.** Its fixed cost -- ~0.23 ms of setup plus
      the extra launches recorded into the solve's graph -- is 0.5-0.9 ms, so any *short* solve
      loses: `min_quad_with_fixed` at 50 % pinned 0.59-0.80x at every size from 576 to 17 689
      unknowns, the heat system (`M - tL`, near-diagonal) 0.7-0.9x, near-converged repeated solves
      (`spd_column_solver_amortized[x50]`) 0.56x, the hole-chain fixed-rim patches 0.84-0.91x. The
      axis is the solve's *length*, not its size (`probes/crossover.py`): at 1 % pinned it wins from
      576 unknowns up (1.3-2.1x). The column-solver defaults stay `"diag"`; the opt-ins are the heat
      Poisson solves, `lscm`, `harmonic` at k=1, `arap` and both `filter_implicit_fairing` solves.
    - **The first run made it the global default and took 40 minutes instead of 3.** The cause is
      the next bullet; the lesson is that a new preconditioner is a per-caller decision, and that
      an A/B over a suite needs a per-process timeout (`/tmp/ab.sh`'s `timeout 1200`).
    - **It needs a symmetric operator, and `filter_laplacian`'s implicit system is not one.** Its
      default operator is the row-normalized uniform Laplacian, asymmetric on boundary rows (0.5 on
      the saddle). Jacobi is symmetric whatever `A` is, so CG gets through in 30 iterations; the
      polynomial in `D⁻¹A` is not, and CG ran to its cap -- 100 s a call. With the tight interval
      below it converged again (10 iterations), and on `saddle_graded` it took the call from 3.3 s
      to 8 ms -- **416x** -- because Jacobi-CG crawls on that asymmetric system. That site is
      deliberately *not* converted: CG on a non-symmetric system has no guarantee under either
      preconditioner, and the right fix there is a symmetric operator, not a faster crawl. **Open
      lead, unmeasured beyond that one row.**
    - **`chebyshev_step` computed the wrong polynomial, in both preconditioners.** The
      semi-iteration's first iterate is `source / theta`; the second step read `source` unscaled as
      its previous iterate. That is still *a* polynomial, so the squared-Laplacian preconditioner
      (`theta ~ 1`, since its interval is `[~0, ~2]`) preconditioned fine and every solve test
      passed. At a Gershgorin bound of 2.2, `theta ~ 1.1`, and the polynomial went **negative
      inside its interval** -- CG took 14 000 iterations where 46 converge. Fixed with a
      `previous_scale` argument; `smooth_region` moved 1.04-1.06x. Caught by evaluating the
      recurrence as a scalar polynomial on a grid of eigenvalues, and pinned by
      `test_chebyshev_preconditioner_applies_the_chebyshev_polynomial`, a closed-form eigenbasis
      oracle: put back, it fails both arms while every solve and count test stays green. **A
      polynomial preconditioner's tests must check the polynomial, not the solve.**
    - **Take the Gershgorin *lower* bound too.** Discs of `D⁻¹A` are centred on 1 with radius
      `r = max_i sum_j |A_ij| / |A_ii|`, so a diagonally dominant operator (`r < 1`) has spectrum in
      `[1 - r, 1 + r]`, far tighter than the `80 / n` lower end a Laplacian needs. `upper` is
      `1 + r` rather than `1 + max(r, 1)`; `r` is floored at `1e-3` so a diagonal operator still has
      an interval.
    - **Scale the rows in the step, not in a copy.** `wps.bsr_copy` was 0.34 ms of a 0.57 ms
      setup; `chebyshev_step`'s `row_scaled` flag multiplies each row's product by `D⁻¹` instead,
      the same arithmetic. And `heat_operators` builds the Poisson polynomial on its **first
      apply**, so `transport_tangent_vectors` / `extend_scalar`, which build the operators and never
      run that solve, stopped paying for it (they had read 0.78-0.90x). `wpl.cg` makes its first
      `M` apply before its captured loop, which is what makes the lazy build legal.
    - **Degree and interval** (`CHEBYSHEV_DEGREE = 12`, `CHEBYSHEV_INTERVAL = 80`): the iteration
      count is monotone in the degree once the recurrence is right, and 12 sits on the flat part of
      wall time from 2.5k to 20k unknowns; a lower end at 5-20 / n costs up to 2x the iterations,
      80-160 / n is flat.
- **Flip rounds: record from round 1, not round 0 and not round 3.** R15-1's premise -- "a
  one-round call pays a recording" -- holds only for a call whose first round flips *nothing*
  (`flip_t_vertices[saddle]`, 1.25x issued). Recording plus instantiation costs more than an issued
  round, so short calls prefer never recording; a first attempt at recording from round 3 (the
  ski-rental break-even) put `isotropic_remesh`'s 2-4-round valence flips at the worst point, three
  issues *and* a barely replayed recording: 0.89-0.92x. Swept on the real callers (`/tmp/flipk.py`,
  interleaved, min of 9), round 0 against round 1 differ by 0.94-1.03x everywhere else, so
  `_FLIP_CAPTURE_FROM_ROUND = 1`.
- **`remove_tunnels`: the cost was a host scan, not the second cut.** R15-7 labelled the would-be
  cut mesh from the face adjacency with the loop edges severed (`kernels/repair.sever_barrier_pairs`,
  a binary search of the host-sorted loop keys in the same launch), so a dependent family is cut
  once -- and that alone moved `handles_64` **1.01x**. A stage profile (`probes/tunnels.py`) put
  57 % of the call in `_independent_loops`, which found each loop's two sides by a NumPy scan of
  the whole face array per loop; the severed pairs *are* those faces, so it reads them from the
  same launch. **`handles_64` 86 -> 36 ms, 2.39x**, `handles_1` 1.00x, byte-identical on CPU over
  eight outputs. What remains: `_cycle_length`'s per-loop readback (30 % of what is left), which
  stays because a host length would reorder the many exact ties a regular fixture has.
- **The captured-run censuses were wrong by a factor that decides the item** (§15.10). Uncaptured
  (`census_plugin.py` with `ATTRIB_NOCAPTURE=1`, which now also switches off the hole-DP and
  flip-round plain graphs): `quadric_decimate[saddle_graded 0.1]` is 21.6 ms device of 43.8, not
  "99 % host"; `fill_min_weight[rim_short]` 5.2 of 6.6, not 0.59. And what the rest *is* differs
  again: timed directly, 42 replays of the recorded decimation pass take 36.7 ms back to back and
  37.3 ms with the per-pass readback between them, so **`capture_while` could recover 0.7 ms (2 %)
  and R15-2 is declined**. The gap is inside the graph: a replayed pass is ~0.88 ms on the device
  against ~0.58 ms of its 109 nodes' own kernel time, ~3 us of gap per node. **A graph replay of
  many small nodes is not free device time; the lever is fewer nodes (fusion), not fewer syncs.**
- **Six-way count pass over the remaining host-bound rows** (agents with disjoint file ownership,
  evidence counts only, CPU byte-identity against `HEAD` for every change; the clock came last):
    - `triangulate_point_cloud`: the candidate grouping is one stable sort emitting runs of 2-3 in
      key order, which *is* `resolve_duplicated_faces`' output order, so
      `_clean_reconstruction(deduplicate=False)` skips that stage -- every face reaching it has a
      distinct vertex set. bunny: allocs 178 -> 131, launches 131 -> 112, readbacks 28 -> 22.
    - `cluster_decimate`: `unique_rows` whose scatter and gather fed only a `.shape[0]`, and a
      `submesh` + `unique_faces` tail, replaced by a remap/compact chain: 23 -> 17 launches, 44 ->
      33 allocs, 7 -> 5 readbacks, 13 -> 7 copies; 1.09-1.36x on every row including `lucy`.
      **The first version of that chain was a 0.19x regression on `lucy` that the counts could not
      show**: it deduplicated *every* face, padding the collapsed ones into a class of their own,
      where the old tail deduplicated the survivors -- and at a coarse voxel size nearly all of
      `lucy`'s 28 M faces collapse, so a hash and a search over the whole input replaced one over
      the output. Launch, allocation and readback counts were all *lower*. **A count census prices
      the host; it is blind to work that grows with the data, so a count-only change still needs a
      clock at the largest mesh before it ships.** Now compact-then-`unique_faces`, which is also
      exactly the old order.
    - `subdivide_region_to_size`: R15-6's `_RegionTopology` rebuild did not exist; the per-pass
      repeat was a whole-mesh `edges_unique` beside a `_FlipTopology` whose radix sort already
      grouped the same keys. `_FlipTopology.edges_unique()` reads them off that sort, and an issued
      flip round regroups only if it flipped: sphere_small 65 -> 43 launches, 105 -> 72 allocs.
    - `fill_min_weight` and everything on its engine: the rim pass writes the rim positions and
      keys it already loads, the traceback writes into the output's tail and reads one shortfall
      scalar: 59 -> 52 allocs, 9 -> 8 readbacks, 16 -> 12 copies. **Not taken**: sizing
      `_EdgeTable` from `len(vertices)` -- §16.4's deliberate negative-index guard, re-found as a
      "redundant" reduction and reverted.
    - `is_watertight` 7 -> 5 readbacks (an order-free share count on `unique_1d`'s own hash table,
      now `grouping.hashed_occurrence_counts`, instead of compact-sort-reduce; `is_vertex_manifold`
      gained `n_vertices=`), `delete_region_keep_boundary` 8 -> 7 (one-scan `submesh`).
    - `remove_degree3_vertices`: the MIS rewrite (R15-A3) is declined -- the input's degree-3
      candidates share no edge on any benchmark mesh, so pass 0 already takes them all, and later
      passes remove vertices earlier removals *created*. What was waste was the final pass that
      finds nothing: the replacement kernel now counts the rim vertices it turns into candidates,
      and a zero stops the loop (bunny 24 -> 16 launches, 35 -> 24 allocs).
    - `fix_self_intersections(local)`: the per-round dilation is two face-mask hops rather than an
      `edges_unique` rebuild (-10 allocs, -1 readback a round). Stage profile: 18.6 of 23.8 ms is
      `refine_and_smooth_region`, of which `_solve_region_smooth` 8.1 ms (the CG 4.5) and
      `subdivide_region_to_size` 6.7. **The single-use captured CG there is not waste**: forcing
      the host-check path at that site is 1.7-2.2x *slower* at every cadence.
    - ICP point-to-plane (R15-A2): 2-9 iterations at the default threshold, and the benchmark pins
      10. A twist-norm stop cannot be shown on sphere fixtures, whose rotation is a free gauge, and
      would change a public contract; declined. Per-iteration allocations went to zero instead
      (cloud 47 -> 17 per call). The Tukey early stop is fixed in the second batch below.
- **`remove_tunnels` was a split->pack round trip plus per-loop readbacks.** `shorten_loop` returns
  ~128 views that `remove_tunnels` measured one gather and one readback at a time and then read
  back again, loop by loop, in four helpers. `kernels/polyline.packed_closed_loop_lengths` measures
  every loop in one launch **bit-identically to `polyline_length`** -- one block per loop, the same
  lane stride and `wp.tile_sum` fold, which for a loop of at most `ITEMS_PER_BLOCK_1D` segments is
  exactly that function's single block -- and one readback of the packed loops feeds every host
  step. Exactness is the whole constraint: the sort keeps equal lengths in basis order and the
  regular `handles_64` has many, so any other summation order changes which loops are cut.
  `handles_64` 86 -> 23 ms by probe, byte-identical on both devices over eight outputs.
- **Refuted on reading, before building**: `quadric_decimate`'s pass allocates dozens of buffers,
  not only `array_scan`'s scratch, so an allocation-free scan alone could not have opened
  `capture_while` to it anyway. **R15-8**: every in-tree `split` consumer is a public function whose
  documented return *is* the list, and each has a packed sibling (`boundary_loops_batched`,
  `query_ball_with_offsets`, `split_batched`); nothing internal iterates a split form except
  `remove_tunnels` (above) and the stitch family (§16.12, deliberate). **R15-A4** stays declined
  (§16.8): only a direct factorization matches potpourri3d's amortized row.
- **`filter_laplacian(implicit_time_integration=True)` was wrong on every open mesh at larger
  `lamb`, and CG was the cause.** The uniform operator is built from directed `mesh.edges`
  (trimesh's own convention, `laplacian_calculation(equal_weight=True)`), so on a mesh with a
  boundary `(1 + lamb) I - lamb L` is **not symmetric** -- 1 056 asymmetric entries on `saddle`,
  768 on `hemisphere`. CG on it matched trimesh at `lamb = 0.5` by luck and returned vertices
  **1.3e4 off on `hemisphere` and 1.3e7 off on `saddle_small` at `lamb = 5`**, and took 3.3 s a
  call on `saddle_graded`; the parity test ran on the closed icosahedron only. Now a fixed-point
  iteration `x' = (b + lamb L x) / (1 + lamb)`: strictly diagonally dominant, so it contracts by
  `q = lamb ||L||_inf / (1 + lamb)` whatever the symmetry, the step count bounding the error at
  `CG_TOLERANCE` is known before the first launch (22 at `lamb = 0.5`), each step is one fused
  `vec3d` launch with no reduction and no readback, and a pass is recorded once and replayed.
  Matches trimesh to 1e-7 closed and open at `lamb = 0.5` and `5`; past 400 steps (`lamb` above
  ~16) or for an operator that is not a contraction it solves the assembled system with
  `wpl.bicgstab` instead (4e-6 at `lamb = 50`). `test_filter_laplacian_implicit_on_open_meshes`
  covers both arms on `hemisphere` / `half_torus`. **Behaviour change**: an unreferenced vertex
  used to shrink by `1 / (1 + lamb)` per implicit pass (its row of the assembled system was the
  diagonal alone); it now stays put, as `operator_row` already made it on the explicit path.
- **The inotify shim** (see §6), compiled with `gcc -shared -fPIC -o fakewatch.so fakewatch.c -ldl`
  and run as `LD_PRELOAD=./fakewatch.so zensical build --strict`:

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
- **Second batch (same day, same method).**
    - **`quadric_decimate` 2.0-4.2x** (lucy 0.1 5.9 s -> 1.4 s), byte-identical on CPU. The pass
      graph went from ~306 nodes to ~87, but **the dominant saving was the six per-round radix
      sorts, not the node gaps**: half of a replay's busy time. A winner's rank in the pass's cost
      order is now read off two bitmasks (popcount plus a word scan), exact against the old stable
      sort of `winner ? cost : +inf` including `inf`/`NaN` costs, and a round is 7 kernels. Every
      scan in the pass is a private allocation-free chunked scan (`remesh._ExclusiveScan`), which
      is what a `capture_while` body needs and `wp.utils.array_scan` is not -- a general
      candidate for `kernels/array.py`, and the thing the flip rounds' plain-graph fallback waits
      on. A pass is re-recorded at the live width once it has halved and freed 250 000 faces of
      width (1.10-1.21x on the scan meshes, a loss on the small saddle, hence the gate).
    - **Tukey ICP stopped after 2 iterations at 4.8 degrees of rotation error** with an explicit
      `robust_scale` smaller than the starting residuals (how the benchmark and Open3D callers
      pass one): the loop tested `sum w r^2`, which a redescending kernel makes *rise* while points
      re-enter it. It now tests the biweight loss `2 rho`, monotone in `|r|`; converges to 0.01
      degrees, and `none` / `huber` / MAD-scale Tukey poses are bit-identical.
      `test_icp_point_to_plane_tukey_converges_from_outside_its_kernel` fails on the old loop.
      **And `cost` now scores the returned transform**: it had been the objective at the pose the
      last step was solved *from*, one step behind `matrix` -- at one iteration the *starting*
      pose's error, 130x the returned pose's, and `inf` at `max_iterations=0`. One more
      correspondence pass and accumulation after the loop (merged with the pending step for a mesh
      target) fixes it, at 0.92-0.98x on the benchmark rows; `matrix` and `transformed` are
      bit-identical. **This is Open3D's convention**, read from `RegistrationICP`: it returns
      `GetRegistrationResultAndCorrespondences` at the final transformation, i.e. correspondences
      searched again at the returned pose, and at `max_iteration=0` it scores the initial one.
      `test_icp_point_to_plane_cost_matches_open3d_evaluation` pins it with Open3D's own
      `evaluate_registration` + point-to-plane `compute_rmse` (`cost == rmse^2 * n`); the lagging
      cost fails its 0/1/3-iteration arms. Open3D's *reported* quantity differs (a point-to-point
      `inlier_rmse` plus `fitness`, whatever the estimator), which is why the test scores through
      the estimator rather than reading `inlier_rmse`. `icp` (point-to-point) keeps trimesh's
      convention -- the fit residual of the returned matrix against the correspondences it was
      fitted to -- which its trimesh parity pins. **Both are "the returned pose"; they differ in
      which correspondences, and each follows its reference. Neither is a bug to align.**
    - **`query_nearest(out=)` declined at 0.97x**: the two `(m, 1)` views a rank-1 `out` needs per
      call plus the checks cost more than the allocation they replace. `registration` launches the
      k=1 kernel directly, guarded by an equality test.
    - **Face-hop vertex morphology**: `expand_vertex_mask` / `shrink_vertex_mask` mark all three
      corners of any face with a selected corner, 2.3-23x without an edge table and flat-or-faster
      *with* one, so their public `unique_edges=` keyword was removed.
    - `face_adjacency(edges_paired=True)`: every edge on exactly two faces makes the adjacency the
      key sort's permutation read two to a row (`is_watertight` 5 -> 4 readbacks);
      `resolve_voxel_grid(return_cell_bound=True)` lets `cluster_decimate` skip the cell hash's
      validating reduction (1.09x); `repair.remove_degenerate_and_non_manifold_faces` filters in
      the input's numbering and compacts once, which every `_clean_reconstruction` caller now uses
      (1.28x on the cleanup); `refine_and_smooth_region` builds one `edges_unique` for the
      boundary mask, both regions and `exclude_fully_selected_components`, where it built three.


### 16.16 Benchmark round 16 (2026-09-24)

Items from `plans/benchmark-round-11.md` (round 16), each timed against a detached `a1c0f29`
worktree through `plans/baseshim.py` (harness, interleaved, min over 2-4 rounds), iteration counts
from `cg_trace_plugin.py`, byte-identity on the CPU oracle where the schedule was the change.
Probes are `plans/benchmark-round-16-data/probes/r16_*.py`; the iteration traces of both arms
are `cgtrace_r16_{base,new}.txt` there.

- **R16-1: every scalar `float64` solve runs `_BatchedCg`, and a solve keeps its state per
  operator.** §12.7's per-call recording of `wpl.cg` was the largest single-solve cost: the
  column solvers now take `_BatchedCg` at one column too, and `solve_spd` does whenever its
  preconditioner is `None` or one `linalg` built *for that matrix* (`jacobi_preconditioner`, new
  and public, and `chebyshev_preconditioner`, both tagged with a weakref to their matrix); any
  other `LinearOperator` still goes to Warp. `_cached_solver` keeps one state per `(operator,
  configuration)` in a `weakref.WeakKeyDictionary`, so a hoisted operator replays its recorded
  loop. `heat_geodesic[amortized]` **2.0-3.1x**, `[full]` 1.2-1.5x, `transport_tangent_vectors
  [amortized]` 1.2x (2.3-2.6x with the block path below), `solve_spd_columns[every]` 1.65x.
    - **The state must not hold its own key.** A `_BatchedCg` holding the matrix keeps the weak
      entry alive for ever, and so does a `_JacobiChebyshev` or a multigrid level. The cached
      state is built over `_storage_alias(matrix)` -- a second `BsrMatrix` sharing the arrays --
      which holds every pointer the graph recorded and not the key. The key carries the arrays'
      identities, so a matrix whose storage is replaced gets a new state. `test_solver_cache_
      does_not_outlive_its_operator` pins the lifetime with `gc.collect()`.
    - **The graph writes `x` through a pointer taken at record time**, so a state that outlives
      one call owns its solution buffer and copies in and out (`_BatchedCg.solve`); the right-hand
      side is read outside the graph. Returned device arrays alias the state and are overwritten
      by the next solve against the operator.
    - **Values rewritten in place are not detected, and need not be for correctness**: the mat-vec
      reads them live, so only the preconditioner goes stale. That is a rate change for Jacobi and
      is used deliberately by `smooth_region_boundary` (below); a Chebyshev or multigrid state with
      a stale interval is a definiteness risk, so no in-repo caller rewrites under one.
    - **A `wp.mat22d` operator is solved as its scalar expansion** (`_scalar_expansion`, cached per
      operator; `kernels/linalg.expand_block_csr_2x2`), with the `wp.vec2d` operands viewed as
      `float64`. Exact, because Warp's blocked Jacobi (`_extract_inverse_diagonal_blocked`)
      inverts each diagonal block's diagonal *coefficients*, which is scalar Jacobi on the
      expansion. `transport_tangent_vectors[amortized]` **2.3-2.6x**, `log_map` 1.2-1.3x.
- **R16-2: a round is two launches -- the Chronopoulos-Gear iteration.** Folding the dots'
  second stage into the consumer blocks (the plan's "three nodes") took a round from **17 to 16
  us**, because each fold is a block-wide `tile_sum` of ~0.4 us on the critical path and two
  chained reductions stay two. Chronopoulos & Gear (1989) recover `alpha` from `r.u` and `w.u`
  (`w = A u`) and carry `s = A p` by recurrence, so one launch does the mat-vec with *all three*
  dots in one `wp.vec3d` reduction (`cg_matvec_dots`) and one does the update with the Jacobi apply
  (`cg_update`). **13 us a round**, against a floor of ~4.3 us for a bare `capture_while` round
  plus ~1 us per replayed node (probed: the conditional test is the largest single part).
    - **`wp.tile(v, preserve_type=True)` reduces a `wp.vec2d` / `vec3d` in one pass**; plain
      `wp.tile(v)` decomposes the vector (§13.2) and `tile_sum` then sums its components together.
    - **Narrower blocks are worse, not better**: at `block_dim` 32 / 64 / 128 / 256 a round costs
      25 / 17 / 13 / 13 us. Each lane then walks several rows serially, and the row walk -- not the
      reduction -- is the latency.
    - **Iteration counts equal standard CG's on 72 of 82 benchmark solves**, the rest within 1.8 %
      either way (2 956 -> 3 009 on the graded saddle's 3 000-round solve, 386 -> 371 on `lscm`).
      No stability loss was seen on any fixture.
    - The stopping test comes first in `cg_update` (the triple describes the `r` the update starts
      from), so the loop runs one detect-only round past convergence; the reported count is the
      rounds that stepped. The recurrence scalars are double-buffered (`*_new` written by the
      update, carried to `*_old` by the next mat-vec), because every block of the update reads them.
    - Setup is two launches too (`cg_initial` -- `r = b - A x`, `u`, `p = s = 0` and `||b||^2`'s
      partials -- then `cg_seed`), where it was `n_columns` copies, a norm, a finalize, a
      tolerance, a mat-vec, a scaling, two copies and an H2D `assign`.
    - Under the Jacobi-Chebyshev polynomial the update writes `D^-1 r` straight into the
      polynomial's input, dropping its scaling launch: `lscm` / `heat_geodesic_conditioning` /
      `log_map` 1.1-1.3x, `arap` 1.05-1.24x.
- **R16-3: `preconditioner="adaptive"`** -- Jacobi under `CG_CHEBYSHEV_PROBE_ITERATIONS = 150`,
  escalating warm-started to Chebyshev. `min_quad_with_fixed`'s default:
  `[saddle_graded pin1pct]` **39.0 -> 14.6 ms (2.67x, now a win against igl's 26.7)**, pin50pct
  flat. **Refuted for `arap`**: its warm-started steps are long, not short, so every step paid
  the probe and a readback -- 0.37-0.63x. `arap` keeps `"chebyshev"` unconditionally, and its
  known `hemisphere 10` loss is gone anyway (1.24x from R16-2).
- **R16-4: `harmonic(k=2)` preconditioned as a squared Laplacian -- and a bug in that
  preconditioner.** `L M^-1 L` is `L D^-2 L` with `D = sqrt(M)`, so
  `squared_laplacian_preconditioner(L_ff, sqrt(M_f))` fits with no new mechanism
  (`parametrization._solve_biharmonic`). It **diverged** at first: the interval's upper end was
  `1 + max(dominance(L), 1)`, which is `D^-1 L`'s Gershgorin bound **only when `D = diag(L)`** --
  true of `smooth_region`'s umbrella weights, false of any other positive `D`, which its docstring
  allowed. The bound is now `max_i sum_j |L_ij| / D_i` (`kernels/linalg.scaled_row_abs_sums`) with
  the lower end scaled with it; `smooth_region`'s counts move by 1-4 rounds. k=2: 179 / 381 / 265
  multigrid rounds -> 52 / 160 / 224, **2.2-3.7x** (`saddle_small` 34 -> 9.3 ms against igl's
  27.5).
- **R16-5 / R16-1(c): `smooth_region_boundary` assembles the band, not the mesh.** The free set
  and the pattern are fixed across passes, so the first pass extracts the band's Dirichlet system
  and every later one rewrites its values and right-hand side from the new half-cotangents
  (`kernels/smoothing.band_dirichlet_values`); keeping the operator object also keeps the solver
  state, whose graph the later passes replay. **2.3-2.5x** on the harness rows (13.4 -> 5.3 ms on
  `bunny`), positions within
  9e-8 on CUDA and **bit-identical on CPU**; `test_smooth_region_boundary_later_passes_match_a_
  rebuild` fails under a sign mutation of the pinned term.
- **R16-6**: `subdivide_region_to_size`'s per-pass region-edge scatter and long-edge test are one
  face-corner launch over a zeroed mask (`mark_long_region_edges`), one launch and one allocation
  fewer a pass. The rest of its cost was not re-attributed this round.
- **Order effects again**: `extend_scalar` read 0.91x inside the heat module's run and 1.08-1.10x
  as its own selection -- the same shape as round 16's `laplacian_smoothing_loss` note. Re-run a
  losing group alone before acting on it.
- **`reconstruction`'s two solves no longer reach `wpl.cg` either -- and "it is `float32`" was
  not a reason to leave them.** The CG kernels are generic over the vectors' *storage* precision
  (`OverloadTable`s `CG_INITIAL` / `CG_MATVEC_DOTS` / `CG_ROUND_DOTS` / `CG_UPDATE` at `float32` and
  `float64`); the cross-block dot folds, `alpha` and `beta` are always `float64`. Storage stays `float32` on
  purpose: the dense grid's top level is 17 M nodes at depth 8 and 134 M at 9, bound by bytes, and
  `float64` vectors would double them. The dense grid is matrix-free (a CSR of the 7-point stencil
  is more memory than the solve), so it has its own mat-vec + dots and initial-residual kernels in
  `kernels/reconstruction.py`, sharing `screened_laplacian_row`, `cg_round_terms`,
  `cg_publish_round_dots` and `cg_update` with the CSR path, driven by `_device.run_device_loop`.
  **`screened_poisson(method="dense")` 1.64-1.68x at depth 7 (19.4 -> 11.7 ms) and 1.30x at depth 9
  (940 -> 722)**; per level 1.8 / 2.4 / 4.9 / 77 ms against Warp's 2.8 / 4.3 / 9.4 / 99. `method="adaptive"` is
  flat (0.99-1.01x): its `warp.fem` matrix's mat-vec is the round, and it is now Warp's own.
  Getting there took four measured corrections, each a trap for the next kernel of this shape:
    - **Deriving a scalar from device values in every thread is `float64` division at scale.**
      `cg_update` read `alpha` / `beta` off the fold and divided per thread; on this GeForce part
      FP64 runs at a small fraction of FP32, and against the same kernel reading constants it was
      **0.53 -> 1.0 ms** at 17 M entries. Past `CG_FOLD_MAX_BLOCKS` a `cg_coefficients` launch now
      derives them once per column (it replaces the old finalize, so no launch is added).
    - **The block count is the reduction's cost** (§2.2, §13.2 again): 66 000 one-tile blocks ran
      the stencil mat-vec with its dots at 190 us a round; `cg_layout` spans a long column's blocks
      over powers of two of tiles to land near `CG_TARGET_BLOCKS = 2048` (80 us). **But only the
      reducing kernel wants it** -- the same span made the pure-stream update 1.4x slower, so the
      update always launches one tile a block.
    - **On L2-resident levels the element arithmetic is the cost, the reductions' included.** At
      2.1 M nodes (8.6 MB a vector against 96 MB of L2) widening every entry to `float64` made the
      level slower than Warp's (10.9 against 9.4 ms). The element updates run at the storage
      precision -- what `wpl.cg` does -- and so do the in-block dot sums (`cg_round_terms`); only
      the few thousand block partials are folded in `float64` (`cg_widen`) and the scalars derived
      there. Per level on `bunny`, depth 8: 1.8 / 2.4 / 4.9 / 77 ms (`float64` in-block sums: 2.0 /
      3.2 / 6.6 / 79). On `float64` storage every conversion is the identity, so the mesh solves
      are unchanged bit for bit.
    - **`float64` reductions buy no accuracy on a `float32` system; `float32` storage sets the
      floor.** True relative residual `||b - Ax|| / ||b||`, recomputed in `float64` on the host,
      identical to 3-4 digits between `float32` in-block sums, `float64` sums and Warp's all-
      `float32` `cg`: at the shipped 100-iteration cap (4.2e-4 / 2.2e-3 / 2.9e-3 / 5.0e-3 by level,
      every level at the cap) and with the cap lifted (8.90e-5 at `tol=1e-4`; at `tol=1e-6` all
      three stall at the same 4.5e-6 / 9.3e-6 / 2.0e-5, the `float32` storage limit). Revisit only
      for a `float32` system whose tolerance is below that floor.
    - **A lane-per-row mat-vec fused with a block reduction loses from ~16 entries a row**: the
      barrier holds every lane until the block's longest row finishes. 400 000-row sweep, fused
      against `bsr_mv`'s lane-per-row kernel: 0.042 / 0.064 ms at 8 a row, tie at 16, 0.22 / 0.13
      at 32, 0.82 / 0.64 at 128 (and `bsr_mv`'s 64-lane tiled kernel 0.31 there). So a long column
      of rows over `CG_HEAVY_ROW_ENTRIES = 16` calls `bsr_mv` and reduces in `cg_round_dots`; a
      *short* one keeps the fused launch even with heavy rows, because below the fold threshold a
      round is launch-bound (`smooth_region`'s 20-a-row, three-column systems read 0.91-0.94x on
      the split path). **Decide on the true count, never `nnz`**: the `warp.fem` system's `nnz` was
      197 M against a true 44 M, and `bsr_mv`'s own tile heuristic reads that capacity.
  `solve_spd(check_every=0)` also stopped warning on the CPU device, where the host-checked
  fallback does have the scalars: the documented contract is that `check_every=0` never warns, and
  its caller there runs a fixed budget on purpose.
- **The capped screened-Poisson solve was the problem the warning reported; silencing the warning
  was not the fix.** Every level of both backends stopped at `solver_iterations = 100` with the
  default `solver_tolerance = 1e-6` unmet -- which `float32` storage cannot reach at all (true
  residual floors of 4.8e-6 / 1.0e-5 / 2.2e-5 / 4.0e-5 at 33^3 / 65^3 / 129^3 / 257^3 under
  Jacobi, 1.2e-6 on the `warp.fem` system). At the cap the dense fine levels were two orders of
  magnitude short (5.0e-3 at 257^3), and the capped surface differed from the converged one by
  **1.0e-3 of the bounding-box diagonal** on average -- the size of the reconstruction's own fit
  error -- where converging Jacobi-CG needed 800 rounds. Fixed on three fronts:
    - **A geometric multigrid V-cycle preconditions the dense solve** (`reconstruction.
      _PoissonMultigrid`; kernels `poisson_mg_*`): the nested node grids, level `l`'s operator
      `2 ** l * L_l + screen * (P^T)^l W` (the Galerkin product's 7-point rediscretization -- its
      smooth-mode ratio to `P^T L P` tends to 2 in 3-D, measured 1.57 already at 9^3), restriction
      `P^T`, one damped-Jacobi sweep at 6/7 a side, four on a 3^3 coarsest grid; `float32`,
      matrix-free, symmetric. **13-16 iterations to `1e-5` at every level, whatever the
      resolution**, true residual following (6.1e-6 at 257^3, *below* the Jacobi floor); the 257^3
      level converged in 24 ms against 77 ms capped-and-unconverged and 471 ms for Jacobi to reach
      even `1e-4`. `screened_poisson[dense-9]` **940 -> 208-246 ms (3.8-4.5x) and converged**;
      `[dense-7]` 1.34-1.46x (a little below the capped `float32` run's 1.64x: the small levels
      are launch-bound and a cycle is five launches a level). Sweeps of two, damping 2/3 or eight
      coarsest sweeps each cost 5-15 % more for no fewer iterations.
    - **`solver_tolerance` defaults to `1e-5`**, above both backends' `float32` floors (the
      multigrid path floors near 2-6e-6); the adaptive backend then converges in 90 of its 100
      iterations (it needed 106 for `1e-6`, which was exactly the CPU warning).
    - **Non-convergence warns on both backends and both devices** (`Warns` section): the dense
      path reads its iteration count once per level (and the residual only on a level that used
      its whole budget), the adaptive path calls `solve_spd` at its default host-scalar cadence.
      `test_poisson_dense_solve_converges_in_few_iterations` promotes warnings to errors at
      `solver_iterations=30` and expects one at 2; swapping the V-cycle for Jacobi fails it on
      every level.
- **`float32` CG for the mesh solves: measured, and it is the tolerance, not the precision, that
  moves the clock.** Jacobi-CG on `icosphere` heat (`M - tL`) and Poisson (`-L`) systems at equal
  tolerance takes the *same* iteration count in either precision, and `float32` is 1.03-1.15x
  faster up to 41 k unknowns (launch-bound) and 1.3-1.4x at 164 k -- while loosening `1e-10` to
  `1e-5` alone saves 1.5-2x in either precision. The solution error then tracks the tolerance
  (~1e-5 relative either way). What `float32` cannot do is the heat method, next.
    - **So it is not a free swap, and it is not taken for the mesh solves.** `float32` storage
      floors the relative residual near `1e-6` (§16.16 above), so it cannot run at the shipped
      `1e-8` / `1e-10` at all, and the switch is really "loosen to `1e-6` and narrow": 100x the
      solution error for the gain. Measured on 1 %-pinned cotangent Dirichlet systems (636 to
      40 553 unknowns, `/tmp`-probe `f32probe.py`): Jacobi at `1e-6` is 1.10-1.17x faster in `float32`
      than in `float64`, identical iteration counts and error -- but `float64` Jacobi-Chebyshev at
      the *shipped* `1e-8` is faster than either (1.2-2.6 ms against 3.1 at 40 k) and 100x more
      accurate, and the polynomial, the V-cycles and `_scalar_expansion` are `float64`-only. The
      lever on these solves is the preconditioner, not the width.

- **`heat_geodesic` was wrong far from its sources on any mesh past a dozen rings -- pre-existing
  (identical at `a1c0f29`), now fixed, and the same defect sat under every heat-diffusion entry
  point.** On a unit `icosphere(5)` from one source its worst error against the great-circle
  distance was **2.34** (mean 0.81), where igl's heat method on the same mesh is 0.019. It is not
  the tolerance's size: a CG iterate after `k` rounds is a degree-`k` polynomial in the operator
  applied to the source indicator, so it is **exactly zero more than `k` rings away**, and the heat
  solve (`M - tL`, well conditioned) met its residual tolerance at ~30 rounds whatever the mesh --
  91.6 % of `icosphere(5)`'s vertices received no heat, so the normalized gradient there was noise.
  The suite never saw it because every igl comparison ran on `icosphere(3)` or smaller.
    - **The fix is a stopping rule, not a tolerance** (`heat._diffuse`): Jacobi-CG in chunks of 64
      rounds at `tol=0`, one tiled launch a chunk (`kernels/heat.heat_chunk_change`) folding the
      largest per-entry change relative to the entry and the count of non-zero entries, one 8-byte
      readback; stop once the reached count stops growing *and* the change is under `1e-6`, or
      three chunks past full reach -- the far field then needs 80-150 rounds past reach to settle,
      and entries near `1e-300` never settle relative to themselves. `extend_scalar`,
      `transport_tangent_vectors`, `log_map`, `heat_signed_distance` and `diffuse_tangent_field`
      all go through it. A `cg_step_scalars` guard (step only when `r.z != 0` and `p.Ap != 0`)
      keeps `tol=0` from reading `0/0` once a system is solved exactly; `!= 0` rather than `> 0`,
      because a negative-definite `min_quad_with_fixed` system goes through the same kernel.
    - **Against igl** (`test_heat_geodesic_matches_igl_far_from_the_sources`, `icosphere(5)`, one
      and three sources, bound `5e-3` of the range): 0.01-0.13 % on the spheres; 0.8 % `bunny`,
      1.1 % `bunny_decimated`, 2 % `hemisphere` from three sources -- the residue is igl's boundary
      convention, next. `extend_scalar` against potpourri3d is 0.0000 on `icosphere(5)` (mean 2e-4
      on `bunny`); `log_map`'s radius 0.8 % / 2.8 % (`icosphere(5)` / `bunny`) where it was 30 % /
      26 %.
    - **igl averages a Neumann and a Dirichlet heat solve on a boundary mesh; triwarp keeps Neumann
      only, on the measurement.** Against `igl.exact_geodesic` on `half_torus`, Neumann is 0.93 %
      mean / 3.1 % max and igl's average 1.15 % / 4.9 %; on `hemisphere` they tie. potpourri3d and
      pymeshlab are Neumann too, and the averaged field broke the `half_torus` comparisons against
      them (and read *below* the Euclidean distance), so it was built and reverted.
    - **Three kernel-scope precision fixes the far field exposed**, all because a diffused value
      near `1e-300` squares to zero: `predicates.stable_length` / `stable_normalize` divide by the
      largest component first (used by `triangles.face_unit_gradient`, the transport and log-map
      kernels); the transport `resolved` mask compares the diffused magnitude against the
      *diffused indicator* at `_RESOLVED_FRACTION = 1e-4` rather than a global floor (the
      `cave_cube` antipode sits at 1.4e-7 of its own indicator from round-off); and
      `extend_scalar` divides only where the indicator is non-zero (`divide_nonzero`) -- obtuse
      triangles make `bunny`'s indicator negative at two vertices, where the old division read 0.
    - **Cost, against `a1c0f29`**: `heat_geodesic` 0.6-1.15x, `heat_signed_distance` 0.67-1.11x,
      `log_map` 0.62-0.74x, `extend_scalar` 0.33-0.35x, `transport_tangent_vectors` 0.17-0.42x.
      The losses are the solves that used to stop at ~30 rounds and were wrong; correctness is the
      claim, and it is paid for in rounds.
    - **REFUTED -- the Jacobi-Chebyshev polynomial in `_diffuse`.** A polynomial round reaches a
      dozen rings where a Jacobi round reaches one, so it is 1.3-1.8x faster on the spheres
      (`icosphere(6)` transport 12.5 against 20.5 ms) -- and **wrong on `bunny` at every chunk and
      settle setting**: the distance 0.68-0.90 of its range off igl's and the scalar extension
      divergent (errors up to 1e5). Obtuse triangles give the heat system positive off-diagonal
      entries, and the polynomial's interval does not cover the far field's decay. Do not
      re-propose it for the heat step.
    - **Why `float32` is out for the heat method**: the implicit step decays by a near-constant
      factor per ring, so the far field falls below `float32`'s range (~1e-38, 1e-45 subnormal)
      within a few dozen rings and to ~1e-300 on the meshes above; igl's `float64` Cholesky resolves
      those values, and the normalized gradient only needs their *direction*, which `float64`
      carries to the antipode. The Poisson step has no such range problem, but it is one solve.
- **Not reached by the cache, and the cache is not the lever there**: `fix_self_intersections` /
  `refill_region` / `fill_smooth` stay flat (0.97-1.04x). Their region solves assemble a *new*
  system every call (`bsr_mm` of the normal equations over a region that just changed), so no
  per-operator state could be hit even if the `SquaredLaplacianPreconditioner` were keyed. And the
  recording is not what they pay: split synchronized at `fix_self_intersections[tangle_torus_small
  local]` (`plans/benchmark-round-17-data/probes/cg_anatomy.py`), the fixed-rim Jacobi solve is
  0.17 ms recording + 1.18 ms for 52 rounds and the squared-Laplacian one 0.85 ms + 2.33 ms for 21,
  of a 22 ms call -- iterations, with the polynomial's 682 tiny `chebyshev_step` launches 3.8 ms of
  the call's 9.3 ms device time. **R16-6's attribution** of the other 6.3 ms,
  `subdivide_region_to_size`: 4.1 ms is its three flip passes, 8 rounds each at ~0.24 ms a replayed
  round -- device, not launch floor. Of a round, the 64-bit radix sort of every edge key is ~86 us
  (68 us at 32-bit keys, which fit only below 65 536 vertices: ~1.5 % of the call, not taken) and
  `delone_flip_candidates` ~59 us, its `float64` Delone predicate on a GeForce part's FP64 rate
  (narrowing it changes which edges flip; not taken).
- **`smooth_region_boundary`'s rim mask re-derived a shared rule, and paid a sort for it.**
  `_region_rim_vertices` built a mesh-wide `edges_unique` only to label vertex components, then two
  histograms, a `flatnonzero` readback and three gathers to drop fully-free components -- which is
  `selection.exclude_fully_selected_components`' rule, now called instead (§2.4's duplicated
  decision rule). And that function, handed no edges, labels over `faces_to_edges` rather than
  `edges_unique`: the union-find labels each component by its smallest vertex id whatever the
  multiplicity and order of the unions, so the labels are identical without the sort. The rim mask
  0.93 -> < 0.2 ms, `smooth_region_boundary` **1.10-1.12x** (harness), byte-identical on CPU with
  `smooth_region` and `fix_self_intersections` unchanged.
- **Round-16 kernel de-duplication pass.** The V-cycle's zero-start pre-smoothing sweep
  `omega D^-1 b` reads no neighbour, so it now rides in whichever launch produced `b`
  (`poisson_mg_restrict` on coarse levels; `poisson_cg_initial` and `cg_update`'s Jacobi apply
  on the finest, with the stored scaling `omega / A_ii` folded at setup), weight restriction and
  the coarse scaling are one `poisson_mg_coarsen` launch, and the finest level stops allocating
  its unused `b` / `x`: a cycle is four launches a level instead of five, **`dense` 1.1x** at
  depth 6-7 with 18-20 % fewer host launches, byte-identical on CPU (CUDA differs by the splat's
  atomic order, 1 ulp, as baseline against itself). `prolong_grid` / `poisson_mg_prolong_add`
  share `poisson_prolong_node`. `_BatchedCg`'s Jacobi diagonal is one CSR-row launch
  (`JACOBI_INVERSE_DIAGONAL`) in place of `bsr_get_diag` plus a map, and `_JacobiChebyshev`
  reads the diagonal and the Gershgorin ratio from one row walk (`jacobi_dominance_rows`):
  `lscm` 1.07x, the rest flat, byte-identical on CPU.
- **The block reductions are one helper family** (`reduce.block_sum` / `block_min` / `block_max`,
  §13.2): 65 hand-written `wp.tile_*(wp.tile(x))[0]` sites across twelve kernel modules became
  calls, with same-dtype quantities packed into one vector per block (7 reductions to 1 in
  `accumulate_loop_frame`, 10 to 1 in the moment integrals). Byte-identical on CPU across 19
  outputs; flat on the CUDA clock, since every caller is launch-bound. `array.tile_argmin` moved
  to `reduce.block_argmin` (it is a block reduction, and `reduce` imports `array`, so it could not
  call the family where it was). **`block_sum` is differentiable**: the two tiled chamfer kernels
  now use it with a scalar `wp.atomic_add` in place of `tile_atomic_add`, and under `wp.Tape` the
  loss is identical and the gradients match `HEAD`'s to its own run-to-run spread (4e-8-5e-7
  relative, the gradient scatter's atomic order) and the analytic nn-term gradient exactly, at
  1, 63, 64, 65, 1 000 and 100 003 points. The same probe found a Warp adjoint bug in the
  *sliced* chamfer kernels, now fixed (§12.4). The last one-value-per-lane reductions spelled with
  tile intrinsics went too: `_reduce_bool_1d_tiled` captures `block_sum` / `block_max` /
  `block_min` rather than the builtins, and `cg_seed`'s `||b||^2` fold is lane-strided, so the
  CG partials lose their zero-filled tile padding (`wp.empty` at `blocks` wide). What remains is
  genuinely tile-shaped: the `tile_load` reduction factories, `remesh`'s chunked scans and the
  cooperative BVH walk. `wp.tile_atomic_add` has no call site left.
- **The final sweep's sub-0.93x cells were drift**: all in functions that reach no solve, and
  0.91-1.04x re-run as their own selection. Median over 280 triwarp cells in the seven modules,
  0.99x.

### 16.17 Benchmark round 17 (2026-09-25)

Items from `plans/benchmark-round-11.md` (round 17), each timed against a detached `4871268`
worktree (`plans/baseshim.py`, interleaved, min of 2), byte-identity on the CPU oracle.

- **R17-1: the heat diffusions run one continuous solve with a device-side settle test**
  (`linalg.solve_spd_settled`, `_BatchedCg(settle=...)`, kernels `cg_settle_change` /
  `cg_settle_decide`). The recorded loop body is 16 rounds plus two launches that compare the
  iterate with the one 16 rounds earlier and may clear the loop condition; nothing is read back.
  Round counts equal `probes/diffuse_need.txt`'s W=16 column exactly (saddle 304, sphere_med 256,
  sphere_small 112 / 96), and the graded saddle stops at reach + 192 = 336. Restarted 64-round
  chunks never passed the settle test on the vector systems, so this is also an accuracy change:
  outputs move by 1e-15 (spheres) to 4e-9 (bunny extension) relative, transport directions
  bit-identical. With the two items below, the heat module is 1.11-2.16x: `heat_geodesic` 1.19-2.14x,
  `transport_tangent_vectors` 1.37-1.80x, `extend_scalar` 1.24-1.36x, `log_map` 1.24-1.45x.
    - **A batched body is right here and wrong for `CG_CHECK_EVERY`**: a round after the loop's own
      stop takes no step, and the settle check needs a block of rounds anyway.
- **R17-2: small solves run as one launch, one block per column** (`cg_one_block`, gated at
  `CG_ONE_BLOCK_MAX_ROWS = 1024`, Jacobi or the squared-Laplacian polynomial). Measured against an
  already-recorded `_BatchedCg` (its best case): 1.5-2.1x at 642 rows, a loss from ~1 200 rows at
  one column and ~1 900 at three. Iteration counts identical on `fix_self_intersections`' two
  region solves (52, 21).
    - **The single block is `float64`-bound, not barrier-bound.** One SM of this GeForce part runs
      `float64` at 1/64 of `float32`, and a 1 000-row mat-vec is ~13 000 double ops -- ~2.7 us of
      one SM's FP64 pipe, which is what the per-pass time measured. So the squared-Laplacian
      polynomial is applied **in `float32`** inside the one-block kernel (a preconditioner need only
      be a fixed positive-definite map; the operator, residual and answer stay `float64`): that
      solve 3.98 -> 2.16 ms (1.84x), against 3.17 ms with the polynomial in `float64`, iteration
      count unchanged. The batched path keeps `float64` throughout.
    - Harness: `fix_self_intersections[tangle_torus_small local]` 1.14x, `fill_smooth[refined]`
      1.10-1.41x, `refill_region[refined]` 1.05-1.24x; `linalg`'s own rows (all above the gate)
      flat.
- **R17-3**: `fillable_loop_mask`'s chord test runs over face corners (idempotent and symmetric,
  so no `edges_unique`), and the pinch test and table scatter are one launch. Byte-identical on
  both devices, 208 of 407 dragon rims fillable. 2.34-3.22x (`dragon` 2.23 -> 0.69 ms).
- **R17-4**: `heat_geodesic`'s normalization is two launches with no readback
  (`source_and_global_sums`, `shift_and_orient`); `marching_triangles_segments` re-zeros at the
  isovalue as it reads (one map and buffer fewer, byte-identical); `smooth_region_boundary` clones
  the field once, writes its pins and side field in one multi-output map and its band in one
  kernel (byte-identical; `smooth_region_boundary` 1.84-1.99x with the one-block band solves);
  `marching_triangles` 1.05-1.11x; point-to-plane ICP against a cloud gathers correspondences inside the
  accumulation (`gather`), dropping a launch per iteration (CPU byte-identical, flat on the clock: the per-iteration
  readback dominates).
- **`heat_operators` / `vector_heat_operators` write their systems over the Laplacian's own
  pattern** (`shifted_system_values`, `linalg.bsr_with_values`): `cotmatrix` and
  `connection_laplacian` keep every triplet, so every referenced vertex's row holds a diagonal
  and `bsr_diag` + `bsr_axpy`'s pattern merge (plus a second `bsr_axpy` for `-L`) is one launch.
  `heat_operators` 1.42 -> 0.81 ms on `sphere_small`, CPU byte-identical. **The three scalar
  operators now share one pattern.**
- **`jacobi_preconditioner` builds Warp's inverse diagonal on first apply**: every solve that
  recognizes the tag reads the diagonal itself.
- **Chebyshev intervals are fitted on the device** (`kernels/linalg.chebyshev_steps`,
  `chebyshev_step` reads its coefficients from an array): building either polynomial preconditioner
  no longer reads the Gershgorin bound back, which in `heat_geodesic` drained the whole queued
  diffusion before the Poisson state could be recorded. On fresh operators: `heat_geodesic[full]`
  1.26 -> 1.65x at `sphere_small`, `heat_signed_distance[band]` 1.07 -> 1.19x, against the same
  build with the host fit. **Its price is a coefficient load per step**, 2-11 % on long polynomial
  solves when the load sat before the row dot and ~0-3 % after it (`lscm[hemisphere]` 0.97x, the
  one cell still below 1.0). Measured, not reasoned: read the coefficients after the mat-vec.
- **R17-5: the settle solves read a `float32` copy of their operator** (`_BatchedCg`'s
  `narrow_values`; `multigrid.csr_row_dot` now accumulates at `x`'s precision and widens a
  narrower value as it reads it, an identity at one precision). Accuracy probed first, by rounding
  the heat systems' values through `float32` in a `float64` solve -- the same arithmetic: distance
  2.1-2.3e-8 of its range on `icosphere(5)` / `bunny`, transported angles 1.6-9.7e-5 degrees at
  most, resolved masks identical, every far-field reference test green. **Speed is far below the
  bytes model**: 1.06-1.08x at `sphere_large`, flat below it (the plan predicted 1.5-1.8x a round
  from 71 MB of traffic; most of that working set sits in this card's 96 MB L2, so halving the value
  bytes halves less than it looks). Harness: `vector_heat_scale[sphere_large]` 1.26 -> 1.33x,
  `heat_geodesic[sphere_large amortized]` 1.23 -> 1.29x against `4871268`.
    - **DECLINED -- a native `wp.mat22d` operator in place of the scalar expansion.** The mat-vec
      alone, both with `float32` values: 0.97x at `sphere_med`, 1.10x at `sphere_large`, ~2 us of
      a ~50 us round -- not a block conjugate gradient's worth of new kernels.
    - **The one-block gate's per-row schedule has no other axis to sweep.** A warp per row needs
      warp shuffles, and Warp exposes only block-wide tile reductions; the block width, swept at
      128-1024 lanes, is the schedule.

### 16.18 The round-16/17 kernel de-duplication pass (2026-09-25)

Every kernel changed in `a1c0f29..1999468` read against the rest of `kernels/`, with the
Python around it. Timed against a detached `1999468` worktree (probe, alternating processes, min of
3 x 15 calls) and gated on the CPU oracle: **every probed output byte-identical on CPU**
(`heat_geodesic`, both `heat_signed_distance` constraints, `log_map`, `transport_tangent_vectors`,
a one-block `solve_spd`, cloud `icp`, `smooth_region_boundary`, `lscm`). CUDA 0.99-1.10x, the two
cells under 1.0 inside run-to-run noise -- these calls are solve-bound, and what was removed is host
work around the solves.

- **Two copies of one decision rule disagreed, which is the finding worth having.** `cg_close_round`
  decided "did column `c >= 1` step" with `r.r > atol and round <= maxiter`, while the column's own
  blocks (`cg_step_scalars`) also refuse a step when `gamma` or the denominator is exactly zero --
  the round-off floor a zero-tolerance heat solve reaches. Such a column counted as stepping and kept
  the loop alive. It now calls `cg_step_scalars` (§2.4).
- **Named runs**: `cg_advance` (the update, `cg_update` / `cg_one_block`), `cg_threshold`,
  `multigrid.chebyshev_update` (`chebyshev_step` / `one_block_chebyshev`),
  `reduce.commit_block_sum` (seven `block_sum` + per-slot `atomic_add` tails, `base` for
  `homology`'s offset slots; `commit_sum_and_count` is now a call to it),
  `laplacian.face_half_cotangents`, `registration.sample_point`.
- **Readbacks.** `_BatchedCg` keeps its per-column scalars in one `(10, n_columns)` buffer with the
  threshold and `r.r` adjacent and the iteration count beside the loop state, so a host result is
  two reads instead of three and the host cadence's condition read carries the count
  (`host_result`); `_cg_residual_and_tolerance` reads an adjacent pair in one copy (`_read_pair`);
  `_cg_one_block` writes each column's round count into its float results, so a host caller
  allocates no integer buffer and reads once (1.10x on a 600-row solve). The dense Poisson cascade no
  longer reads each level's count before issuing the next -- one read of all levels' counts after
  the cascade -- and allocates its splat and CG vectors as one buffer each (flat at depth 7).
- **Launches and buffers.** `heat_signed_distance`: the unit-field map folded into
  `vertex_field_divergence`, which also accumulates `-div` (no negate pass on either constraint),
  and the `"none"` shift is `heat_geodesic`'s device-side one without orientation -- no gather, no
  readback (1.06x). `log_map`: two maps folded into `log_map_from_angles` (1.04x).
  `transport_tangent_vectors`: the extension's division folded into `transported_and_resolved`.
  `_BatchedCg._initialize` under Jacobi-Chebyshev writes the polynomial's input from `cg_initial`, as
  `cg_update` does every round. `smooth_region_boundary`'s later passes form the band faces'
  cotangents in `band_dirichlet_values` instead of a whole-mesh table per pass (1.05x).
- **`icp_point_to_plane` against a cloud applies its step inside the next nearest search**
  (`neighbors.query_bvh_nearest_after_step`), as the mesh loop already did: one launch and one `n`
  round trip fewer per iteration, **1.03-1.10x** at 20 fixed iterations, byte-identical on CPU. The
  row kernels' deepening loop became one `wp.ref`-row `@wp.func` per bucket (`_bvh_row_search`),
  which the plain kernel and the stepped one both call; `query_nearest` at `k` = 1, 8, 32 measured
  flat. **Two optional arguments on the shared k-NN kernels were built first and measured 0.96-0.97x
  on their other callers at 2 562 queries** -- two marshalled arguments on a launch-bound call -- so
  the step is a variant of its own.
- **`log_map` is not reproducible at the cut locus on CUDA, before or after.** One vertex of
  `icosphere(5)` at radius ~pi differed by 3e-2 between the two builds and by 3e-3 between two runs
  of the new one; the docstring already calls that angle arbitrary. Compare off the antipode.
- **Declined.** Folding the Gershgorin max into the row kernels (`scaled_row_abs_sums`,
  `jacobi_dominance_rows`): saves one launch and an `n` buffer per preconditioner build, but a
  block-folded row walk gives each lane 16 rows where the row-per-thread kernel has one, which at
  large `n` costs more than it saves. **Two launch removals in point-to-point `icp`, both built and
  measured flat** (0.98-1.02x at 2 562 and 40 962 points, 20 fixed iterations), so both reverted:
  reading a cloud target through the nearest index inside the Procrustes kernels instead of a
  gather copy, and ping-ponging the accumulator so each fit's matrix launch zeroes the next fit's
  instead of a memset. The loop is paced by its per-iteration readback, which is what a removed
  launch hides behind. **Measure it at a fixed iteration count** (`threshold=-inf`): at
  `threshold=0.0` the plateau's last-bit jitter stopped the arms after 12-20 iterations at random and
  read as a 0.67x regression.

### 16.19 Benchmark round 18 (2026-09-25)

Items from `plans/benchmark-round-11.md` (round 18), each against a detached `192406c` worktree
(`/tmp/tw18base`, §15.6), byte-identity on the CPU oracle unless stated. Clocks were taken on a box
shared by four concurrent reviewers (min over alternating processes); probes are
`plans/benchmark-round-18-data/probes/r18_*`.

- **R18-1: independent diffusions stack into one settle solve** (`linalg.block_diag`, public,
  cached per set of operators so the stack's state replays; `kernels/linalg.stack_block_diagonal`
  reads each block's count off its own last offset, never `nnz`). `transport_tangent_vectors`
  solves `[vector; heat; heat]`, `log_map` `[vector; heat]` (`heat._distance_from_heat` is the
  Poisson half), one seed launch each. Hoisted operators: transport 1.6-1.9x, `log_map` 1.1-1.5x.
  **A single CG over a block-diagonal stack is not two independent CGs**: `alpha` / `beta` span
  every block (the `_BatchedCg` docstring's "CG on the block system"), so the iterates differ at a
  given round and agree only once settled -- 1e-7 on the saddle and spheres, while on
  `saddle_graded`, which never settles, `log_map`'s distance error against `igl.exact_geodesic`
  moved 9.4 -> 9.7 % mean at the *same* 336 rounds. Stack only systems that settle together.
- **R18-2: a fresh operator of a seen shape takes a pooled state** (`linalg._cached_solver(...,
  pooled=True)`, `_BatchedCg.refresh`, `kernels/linalg.refresh_pooled_operator`). The pooled state
  owns a copy of its operator; one launch copies the next operator in and re-derives the narrowed
  values, the Jacobi diagonal and the Chebyshev ratios (refit on the device), so the recorded graph
  replays. **Refreshed on every use**, not only when the operator changes: an owning state cannot
  see in-place rewrites (`smooth_region_boundary`'s later passes rewrite their band values) and two
  live operators of one shape may alternate. The first operator of a shape keeps an aliasing state,
  so a hoisted operator pays no copy. Only immediate-use callers pool -- `_AdaptiveCg`
  keeps its states across calls and must not. Fresh-operator calls: `heat_geodesic` 1.62x, transport
  1.6-1.9x, `log_map` 1.3-1.6x, **zero `ScopedCapture`s per call** (was 2-3).
  `test_pooled_solver_state_follows_each_operator` bites a no-op refresh and a skipped refit
  (the refit is a rate, so only the step comparison sees it).
- **R18-3: k = 1 nearest through a `wp.Mesh` of collapsed triangles** (public
  `neighbors.mesh_from_points`): `query_nearest(k=1, backend="bvh")` without an accelerator, both
  ICPs' cloud targets and the Chamfer/Hausdorff cloud searches. Distances byte-identical to the BVH
  walk; `bvh_leaf_size=1` is 1.4-1.7x Warp's default for this mesh. **Load-bearing Warp detail:
  `mesh.h`'s sliver cull `|n| / sum|e|^2 < 1e-6` reads NaN on a collapsed triangle, which compares
  false, so it is tested rather than culled** -- `test_mesh_from_points_answers_the_nearest_point`
  fails if a release changes that. `query_nearest_bvh_k1[dragon]` 2.22 -> 0.89 ms; Chamfer on the
  shifted fixture 2.2-3.1x; a coincident dragon pair read 0.81-0.98x, fixed in §16.20.
- **R18-6: point-to-plane ICP's pinned loop runs on the device** (`point_to_plane_round`, three
  launches a round under `_device.run_device_loop`, recorded once per call; iteration 0 on the host
  for the MAD scale). A cloud call 34 -> 10 launches, 12 -> 2 readbacks; 1.4-5.6x with R18-3. A call
  that is weightless at its seed still records a loop of zero rounds: 0.81x, failure path only.
  Point-to-point `icp`'s ten readbacks are the open lead.
- **R18-4: `remove_degree3_vertices` without `vertex_one_rings`**. An interior degree-3 vertex is
  three faces whose opposite edges close a directed 3-cycle, and the cycle *is* the replacement;
  a telescoping `x - y` sum over the opposite edges is an exact "closed link" flag (0 for any
  closed link, non-zero for one boundary fan, also mod 2^32). Selection and emit fuse, one readback
  a pass: 1.43-2.15x. Pinched input is now accepted (never a candidate) rather than raising from the
  ring build. `remove_unreferenced_vertices` in one mark, scan and compaction: 1.29-1.55x.
  - **REFUTED -- a hash-only validate for `halfedge_twins`** (directed-edge uniqueness through
    `hashed_occurrence_counts`): 1.26x on bunny, **0.59x on dragon** -- random atomics into a
    2^23-slot table lose to the radix sort (`r18_twin_validate.txt`). **Specific to that table**:
    round 20's `point_duplicate_mask` hash (`int32` slots at 2n, one CAS a point) beat two chained
    sorts at every size to 4 M points, while its region-boundary hash lost 0.74x at `dragon`
    (§16.23). Probe the table shape; do not generalise either result.
  - Taken instead: `_pair_halfedges` and the two edge-manifold predicates write their keys with
    `adjacency.face_edge_keys` rather than `faces_to_edges` + `hash_indices_rows` -- the fused
    kernel existed and three callers did not use it. `halfedge_twins` 1.13-1.16x byte-identical;
    `is_edge_manifold` / `edge_manifold_mask` 1.04-1.10x.
- **R18-5: the region normal equations assemble directly** (`smoothing`, no `bsr_*` call left in
  either region solve; `SquaredLaplacianPreconditioner.from_factors`). One incidence build per
  region pair; the fixed-rim values and `M^T M` are written over region-own patterns. A
  fixed-rim + smooth pair 38 -> 23 launches, 113 -> 57 allocations, 6 -> 3 readbacks;
  `tangle_torus_small` pair 6.99 -> 4.54 ms, `holes_many` smooth 2.88x. Iteration counts unchanged
  on all 14 captured systems. **`bsr_mm` sums most product entries along the row and the rest in
  its triplet sort's order, which is not stable on CPU, so it is reproducible only to one rounding
  per entry** -- 54 of 1 000 `M^T M` rows differ by one rounding, nothing else moves. A face
  repeating a vertex no longer contributes a self-loop to the unit weights (0.019 on that input).
  Declined: matrix-free `M^T M` in the one-block solve (ceiling under 0.3 ms).
- **R18-7: `triangulate_polygon` clips the 2D ring directly**, one prologue readback, and **a
  single-block ear loop** (`ear_clip_block`, 1 024 lanes, `block_sum` barriers) up to
  `EAR_ONE_BLOCK_MAX = 1024` on CUDA and at every size on CPU: 3.4x at 64 corners, 1.66x at 1 024;
  a loss past the cap (0.39x at 4 096). Removing its two step barriers **hangs** on CUDA. The
  near-collinear ring's faces changed (the orientation test now reads input coordinates rather than
  centred-and-rotated ones -- §12.4's sign-test hazard); both are valid, and the new one covers
  exactly trimesh's region (20 000 / 20 000 samples, both devices), which
  `test_triangulate_polygon_near_collinear` now asserts. No reference pins the diagonals.
  `polyline_radius` one fused pass, 3 -> 1 readbacks, 2.47x. **A per-thread closure predicate beats
  a `dim=1` flag launch on CUDA and loses at 20 000 points on CPU** (0.87x, accepted per §9).
  - The one-block crossover (~1 024 items) is §16.17's CG crossover again: a graph recorded per call
    is most of a small round loop, and one block of barriers undercuts it until the work outgrows
    one SM.
  - **Taken: a one-block RDP for `polyline_simplify`** (`rdp_simplify_block`, the four round
    steps as shared `@wp.func`s the launch form also calls). Keep masks byte-identical on both
    devices; CUDA 2.0-2.3x to 1 024 points, 1.28-1.30x at 4 096, 0.86-0.96x at 8 192, so
    `RDP_ONE_BLOCK_MAX = 4096` -- past the ear loop's 1 024, since an RDP round is four streaming
    passes rather than an O(ring) test per corner. CPU 1.5-5.4x at every size. Deleting one step
    barrier fails `test_simplify_block_and_round_loop_match_reference` on the 1 500-point walk.
  - The ear loop's init and clip writes became `init_ring_slot` / `clip_ear`, which drops check 13's
    `init_ring` / `clip_selected` allowlist entries and `rdp_split_spans`' `span_lo` / `span_hi`
    (§4.5: the extraction wins over the syntactic check).

### 16.20 Round 19: the round-18 leads (2026-09-25)

Against a detached `8f39161` worktree (`/tmp/tw19base`); CPU byte-identity unless stated; clocks
on a box shared by three reviewers. Probes are `plans/benchmark-round-19-data/probes/r19_*`.

- **Point-to-point `icp` runs on the device**: five launches a round (the correspondence pass moves
  each point by the kept transform in registers and searches cloud or mesh by a warp-uniform flag;
  `point_to_point_round` keeps a fit only if it carried weight and runs the host's float64 stop
  test). A pinned 10-iteration call 42 -> 13 launches, 10 -> 1 readbacks; 1.07-1.46x, but a
  1-2-round call is 0.82x -- a recording (~70-85 us host plus ~19 us to launch) pays back after
  about two replayed rounds of 30-55 us gap. REFUTED: issuing K host rounds before recording
  (K = 2 won 4-8 % on 2-round calls and lost up to 37 % on weightless ones).
- **Record while the device runs, then read, then launch** (`_device.record_device_loop`, which
  `run_device_loop` is now a call of). Reading round 0's outcome before recording costs every gated
  call 2-8 %; recording first hides the recording behind round 0 and makes the read free. The
  weightless-at-seed ICP call went 0.83 -> 0.94x of `192406c`; the residual is the recording, which
  only a read before it could skip. A weightless round writing an **identity step** lets the
  closing pass report `inf` with no readback of its own (bit-identical except an exact `-0.0`).
- **The MAD scale runs on the device without compaction**: `+inf` for excluded residuals, one radix
  sort, a prefix search, medians exact against `reduce.median`; 3.1x on the helper, 1.10-1.28x on
  the Tukey/Huber/MAD rows.
- **`metrics` picks the backward search from the forward half's answer** on CUDA past 262 144
  points: the seeded hash grid when the largest forward distance is at most half the density
  radius and at least 75 % of backward queries are some point's forward answer, else the
  collapsed-point mesh. Coincident dragon 1.12-1.32x of `192406c` (the round-18 loss is gone),
  displaced pairs keep the round-18 win (2.0-3.0x) at 0.95-1.01x of `8f39161` -- the one readback
  the choice costs. Two things that decide it: **a small forward maximum does not make a pair
  coincident** (a cloud covering half its partner has one), and the hit fraction separates them
  cleanly (>= 0.89 vs <= 0.5; one `atomic_exch` per point in the same launch); and the readback, not
  the search, is the cost below ~250 k points, because it stops the backward half's issue from
  overlapping the forward half. The grid's best seed is 0.1-0.3 of `knn_initial_radius`.
- **`closed=True` wraps the index** in `polyline_upsample` / `smooth_upsample` / `downsample` /
  `resample` / `point_distance`: no `polyline_close` copy, closure decided per thread; tables sized
  for `n` segments with a zero-length closing slot when the input already repeats its first point
  (`seam_repeats_first`), so every prefix is the `n - 1`-segment one bit for bit on both devices.
  Closed paths 1.4-2.4x on CUDA; `polyline_resample(closed=True)` on an empty input now returns
  empty (it raised). **Evaluate a per-thread predicate before a long loop, not after**: the same
  closure test placed after `distance_to_segments`' 65 536-iteration loop cost 13 % device time
  even when its branch was not taken. The `allclose` tolerances became kernel-scope constants
  (`constants.ALLCLOSE_*_CONSTANT`), removing two launch arguments from every polyline closure
  kernel -- which is what the per-thread test had cost the open paths (0.90-0.97x -> parity).
- **Probe hygiene**: two reviewers overwrote a shared probe helper mid-run and one A/B silently
  loaded the live tree in both arms. Give concurrent reviewers' probe helpers unique names, and
  assert the *submodule's* `__file__` -- the lazy `__init__` means `triwarp.__file__` proves nothing
  about which `triwarp.polyline` was imported.

### 16.21 The round-17/18/19 kernel de-duplication pass (2026-09-26)

Every kernel changed in `1999468..e34051f` read against the rest of `kernels/`, by five reviewers
with disjoint file ownership, counts as evidence and CPU byte-identity against a detached
`e34051f` worktree; every probed output is byte-identical on CPU. Quiet-box clock afterwards
(alternating processes, min of 3), `icosphere(5)` / a 3 000-point ring: `polyline_normal` 1.58x,
`polyline_simplify(closed=True)` 1.32x, `smooth_region_boundary` 1.40x, `marching_triangles`
1.23x, `remove_unreferenced_vertices` 1.16x, `polyline_radius(closed=True)` 1.15x, ICP and the
heat family 1.02-1.04x, Chamfer flat. What generalises:

- **A band-sized question was answered with a mesh-sized matrix.** `smooth_region_boundary`
  assembled `-cotmatrix` for the whole mesh to extract a system over a few hundred rim unknowns;
  its pattern now comes from the vertex-face rings (`band_pattern`, a count and a fill pass of one
  kernel). §16.5's densify-the-whole-mesh shape again.
- **A `_BatchedCg` solve's prologue replays with its loop.** `cg_seed`, the preconditioner's first
  apply and the settle reset are recorded ahead of `capture_while` in the same graph, so a cached
  solve issues only `cg_initial` (which also copies the initial guess in): a Chebyshev `solve_spd`
  13 -> 1 launches, 2.6-2.9x on a short solve. The loop state and count share the float64 scalar
  buffer's head, so a host result is one readback.
- **`heat_operators`' timestep leaves the edge sum and count on the device** and
  `shifted_system_values` forms the mean itself: no readback; `vector_heat_operators` builds its
  lumped mass once. `heat_signed_distance` derives curve segments per thread (binary search into
  the offsets) instead of a host loop over a readback.
- **A compaction is one kernel over an int32 flag scanned in place**, not a bool mask +
  `flatnonzero` + one gather per output buffer (`intersection`'s three segment producers,
  `polyline_simplify`); unused optional outputs pass `None` and test `shape[0]` (a null
  descriptor's shape reads 0 on both devices).
- **`closed=True` wraps the index** in `polyline_simplify` / `polyline_radius` too, and
  `loop_point` (one select) replaces `wrap_index` (two modulos) wherever the index is at most `n`.
- **Both ICP loops share one correspondence search** (`icp_match`, warp-uniform cloud/mesh) and one
  stopping rule (`continue_icp_loop`); the per-correspondence normal gather is gone (normals read
  from the target's table at the index).
- **`reduce.block_barrier`** is the one spelling of a discarded block reduction used as a barrier
  (CG one-block kernels, the stitch DP tile, the RDP and ear-clip block loops).
- Declined, with reasons at the sites: merging `nearest_point_via_mesh` with
  `query_nearest_via_mesh` (a launch argument on a launch-bound k = 1 path, §16.18's 0.96-0.97x),
  merging the Jacobi/dominance kernels behind flags (a shared `jacobi_row` helper instead).

### 16.22 Benchmark round 20 sweep (2026-09-26)

Full sweep at `a2ec2e5` against round 18's at `192406c`: gap 97.2 -> 68.2 ms, 746 wins of 928, no
regression (the five slower cells re-measured at 1.00-1.02x). The top 39 ms of what remains is
floors -- list returns, device-bound traversals, and settle solves bound by their round count. Two
declines, measured (`plans/benchmark-round-20-data/probes/`):

- **Folding a CG round's dots in every block past `CG_FOLD_MAX_BLOCKS = 256` tiles is a loss**,
  though the unfolded path's third node looks like the cost. On transport's stacked 70 756-row
  settle solve (277 tiles, just over the cap) a round is 10.4 us `cg_matvec_dots` + 7.2 us
  `cg_coefficients` + 5.1 us `cg_update`, uncaptured. Raising the cap to 512 / 1 024 or doubling
  the span until the column fits under 256 blocks measured 0.90-0.97x on transport, 0.82-0.85x on
  `transport_scale[sphere_med]`, 0.65x at `[sphere_large]` and 0.84-0.86x on
  `heat_geodesic[sphere_large]`: the redundant per-block fold and per-thread `float64` divisions
  cost more than the node. An empty in-graph launch is 1.1 us; a `float64` block reduction adds
  0.9-1.5 us and a `wp.vec3d` one 2.2-3.5 us -- twice a scalar's, the same as three scalars.
- **The heat Poisson solve's Chebyshev degree is at its optimum.** Degree 2 / 4 / 8 / 12 against
  Jacobi at 2 562, 40 962 and 163 842 rows: 12 wins or ties (degree 8 is 3 % ahead at 2 562), so the
  small `heat_geodesic` row's 231 polynomial launches are this method's floor.

### 16.23 Round 20's items and the pass around them (2026-09-26)

Every R20 item landed, byte-identical on the CPU oracle against a detached `a2ec2e5` worktree
(`/tmp/tw20base`) over ~3 300 outputs in eleven gates; on CUDA the only differences are the ones a
second baseline run also shows (ear-clip face order, the orientation propagation on `boy` /
`mobius`, float atomics in `cluster_accumulate` and the remesh passes). Clock: harness A/B, one
pytest process per module, min of 2 rounds (`plans/benchmark-round-20-data/ab21_*.txt`): median
1.02x over 566 cells, 51 cells at 1.5x or better, and every cell under 0.93x either re-measured
level alone or has identical counts and device time in both arms (next paragraph).

- **R20-1** `extrude_polygon` / `sweep_polygon` wall a full `n - 2` triangulation from the ring
  edges (the derived boundary is the fallback, and still raises `sweep_polygon`'s non-simple-ring
  `ValueError`): **3.1-3.3x / 2.8x**. `_PARAMETRIC_LATTICE_DEVICE_FROM` fell 9 216 -> 625 samples,
  re-probed quiet: host and device tie at 400 samples and the device leads from 900.
- **R20-2** `point_duplicate_mask` is one `atomic_cas` table of `int32` point indices at >= 2n slots,
  bitwise compare, `atomic_min` to the class's first index: 14 launches / 21 allocs / 2 readbacks ->
  2 / 3 / 0, **8-12x to 1M points, 33x at 4M, 6.6x on a 1.1M-point axis lattice**
  (`probes/r20_points_dup_ab.txt`, quiet). `-0.0 == +0.0` and identical NaN rows merge, as before.
  **So §16.19's "hash loses to sort at the large end" is a fact about that table, not a rule**: the
  same round's region-boundary hash lost 0.74x at `dragon` (`probes/r20_region_boundary_quiet.txt`)
  and the sort shipped there. Probe each table shape.
- **R20-3** `crease_edges` 10 / 16 -> 7 / 9 (one `crease_flags` kernel on
  `predicates.vector_angle`, the predicate `face_adjacency_angles` already uses): 1.07-1.40x.
- **R20-4** `boundary_edges` 1.00-1.52x (keys straight from faces, one emit);
  `region_boundary_edges` 9 / 16 / 2 -> 3 / 4 / 1, **2.54x** (one sort of every halfedge key, the
  seam a run of two straddling the region).
- **R20-5** `faces_left_of_contour` hooks faces across twins in the thread -- no dual-edge list,
  cursor, trim or readback, pre-hook kept (`ecl_prehook_pair` / `ecl_hook_pair`, now shared
  `@wp.func`s in `connected_components`): **1.51x**.
- **R20-6** `cluster_decimate` 6 -> 4 readbacks (`unique_faces(max_index=)`, a new public keyword
  that skips the validating reduction; both counts read from one 2-slot buffer the compaction
  kernel publishes into a prefix view of a worst-case buffer): 1.05-1.35x.
- **R20-7** `is_watertight` 21 / 25 / 4 -> 12 / 12 / 2, **1.58-1.65x**; `is_volume` 2.0-2.1x;
  `is_vertex_manifold` / `is_edge_manifold` 1.4x; `face_self_intersecting_mask` 1.40-1.45x (one
  fixed-stride broad phase, no count pass, no readback). **Fusing the narrow phase into the walk
  stays declined, re-measured: 0.91x on `sphere_med`, 0.27x on `tangle_2`.**
- **R20-8** `mesh_collision_pairs` deep 8 / 11 / 2 -> 4 / 5 / 2, `mesh_with_mesh` 9 / 13 / 3 ->
  4 / 6 / 2 (narrow phase inside the segment kernel): 1.0-1.4x. `mesh_with_plane` computes the
  plane dots in-kernel.
- **R20-9** the discarded `counts_to_offsets` total is gone; no other call site in the tree
  discards one (22 checked).

**What generalises from the extras** (each byte-identical, counts at the benchmark's point):

- **`radix_sort_pairs(end_bit=)` is the cheapest lever found this round** (§13.1). Taken where a
  packed key's radix is known: the flip loop's regroup (**88 -> 55 us a replay** on
  `tangle_torus_small`, 134 -> 108 on `tangle_torus`; the regroup itself is gone since, §16.24),
  `sorted_undirected_edge_keys`, the deleted-face keys, the voxel buckets, the candidate-triangle
  keys, and `unique_1d(max_value=)` -- a new public keyword `edges_unique`, `unique_faces`,
  `cluster_decimate`, `split_nonmanifold`'s labels and `submeshes_from_face_groups` pass. The pair
  radix (`INDEX_RADIX_PAIR`) needs all 64 bits, so a function with no vertex bound gains nothing.
- **`adjacency.sorted_face_edge_keys(faces, *, n_vertices=None)`** (new, public) names the run four
  modules repeated: pack straight into the sort's double-width buffer -- no staging copy, which at
  `lucy` is 670 MB -- and sort only the bits a known radix needs. `face_adjacency(edges_paired=)`,
  `halfedge_twins`, `face_connected_component_labels` and `validation`'s sorted keys use it.
- **`face_connected_component_labels` hooks the union-find straight off the sorted keys**
  (`sorted_pair_prehook` / `sorted_pair_hook`): no compaction, no readback, no edge table.
  1.46-2.39x to `happy_buddha`; the first cut wrote a `(3n, 2)` edge table and read **0.80x at
  `lucy`**, the key staging copy was another 0.07, and it is 0.97x there now. A saving that removes
  a readback and adds `O(n)` traffic shrinks with the mesh and can invert: measure the top of the
  axis.
- `cut_along_edges` 14 / 23 / 3 -> 9 / 12 / 2 (union-find over halfedges; the root-flag scan's
  ascending ranks *are* `unique_1d`'s sorted inverse): 1.64-1.99x. `isotropic_remesh` ~15 readbacks
  fewer on the saddles (the collapse pass shares `cluster_decimate`'s compaction): 1.11-1.13x.
  `statistical_outlier_mask` 3 -> 1 readbacks (a `float64` device threshold; can move an ulp,
  identical on 30 masks): 1.26-1.44x. `fit_plane` / `principal_axes` fold the centroid division
  into their consumers. `voxel_down_sample` 2 -> 1 readbacks. `unique_1d` one allocation fewer
  everywhere. `kernels/triangles.sort_face_indices` deleted (no caller left).

**The cells that read slower, and why none is code.** A first A/B ran all fifteen modules in one
pytest process and showed ~24 cells under 0.93x -- `face_adjacency_angles[dragon]` 0.47x twice,
`mean_edge_length[dragon]` 0.61x twice, a dozen more. **Every one of them is level when its module
runs in its own process** (0.96-1.14x); the round's sweep is one process per module for this reason,
and a multi-module A/B inherits the preceding modules' allocator and cache state, so a cell's arm
depends on what ran before it. Of 366 per-module cells, two stayed under 0.93x across three runs
(`radius_outlier_mask[sphere_small 4.0]` 0.86x, `intrinsic_delaunay[sphere_large]` 0.91x); both
have identical launch, allocation and readback counts and identical device time in the census,
and `radius_outlier_mask`'s code is untouched -- order effects inside the module.

**Two traps the pass hit, both caught by re-measuring rather than trusting the report:**

- **A BVH walk's box formed in the kernel from the face corners was 4.5x slower than the identical
  walk reading a stored box** -- same candidates, same registers (37-38), no spills. Two reviewers
  fused the box into their broad phases independently and both were device regressions
  (`face_self_intersecting_mask` 0.27 -> 0.53 ms, `mesh_collision_pairs` deep 0.16 -> 0.29 ms)
  hidden under a host saving. Both now launch `triangles.face_aabb_bounds` first (4 us). The cause
  is not the box: across five spellings of one walk over the same BVH the time ranged 97-523 us
  (`mesh_query_aabb` vs `bvh_query_aabb`, a literal vs a runtime cap), which is nvcc code-generation
  variance rather than a rule (§14.2). **Time a new BVH-walk kernel against its predecessor's
  device time; never assume a fusion that removes a launch is free on the device.**
- **Four of seven reviewers stopped mid-edit on an API spend limit**, and a docstring rewrap one of
  them ran had folded function bodies into two docstrings (`tests/test_creation.py`,
  `benchmarks/test_creation.py`) -- `SyntaxError`, found only when the suite collected. After any
  bulk docstring edit, `ast.parse` every changed file; after an interrupted pass, re-run every
  gate against the final tree, since a reviewer's saved `new` outputs describe a mid-state.

**Platform facts the pass confirmed** (both devices): a kernel-scope `wp.array2d` indexed once
(`table[1]`) is a contiguous rank-1 view that can be handed to a `@wp.func` taking a `wp.array`,
and writes through it land in the parent -- which lets a helper that emits into a flat buffer
(`proximity.mesh_aabb_collect`) fill one row of a `(2, n)` pair table with no expansion pass; a
Python-scope gather (`wp.indexedarray`) passes straight to a kernel parameter annotated
`wp.indexedarray[T]`, removing the gather copy when the consumer only reads; `wp.copy`'s offsets on
a contiguous rank-2 array count scalar elements, not rows; and `.numpy()` on a strided column view
is one readback plus an internal contiguous copy.

### 16.24 The flip loop keeps its adjacency instead of regrouping (2026-09-26)

Round 20's open lead, priced at 11-15 % of `fix_self_intersections` and built on request. Every
`_flip_interior_edges` round used to end in a whole-mesh regroup (a key launch, a radix sort, a
mark, a scan, an emit); now only the first flipping round does, and that regroup also builds the
incremental state every later round -- the recorded ones -- runs on. **Byte-identical on CPU**
against `9612307` over 46 outputs of every flip entry point (`flip_to_delaunay` whole and region,
`flip_by_objective` x3, `isotropic_remesh`, `subdivide_region_to_size`, `refine_region_to_density`,
`flip_t_vertices`, `fix_self_intersections`, `delaunay_triangulation`), also with the rebuild
budget forced to 7 flips; on CUDA the only differences are `isotropic_remesh`'s two run-to-run
outcomes, which the baseline alternates between too. Harness, one process per module, min of 3:
`delaunay_triangulation` 1.26-1.44x, `flip_by_objective` 1.12-1.37x, `flip_to_delaunay` 1.20-1.22x,
`fix_self_intersections(local)` 1.10x / 1.16x, `refill_region[lucy refined]` 1.38x,
`fill_smooth[rim_short refined]` 1.14x; `subdivide_region_to_size[None]` at parity (0.97-0.99x).

**Three things make it byte-identical, and the first is the one that is not obvious:**

- **The claims rank candidates by edge key, not row index** (`kernel_remesh.flip_priority`,
  `uint64` `atomic_min`, which works on both devices). A regroup emits rows in ascending key order,
  so the smallest row index *was* the smallest key: the winners are the same whatever order the
  rows are in, which is what frees each edge to keep its row.
- **Rows are rebuilt from maintained halfedges, not patched.** `commit_flips` keeps a
  halfedge -> row and a row -> halfedges map current (each side edge of a flipped quad keeps its row
  and moves to a new halfedge of one of the two rewritten faces), and `refresh_flip_rows` rewrites
  every row through the same `write_flip_row` the emit uses. Patching a row's face and apex columns
  in place instead races: a side edge's two faces can belong to two different committing flips.
  Moving a halfedge is race-free because each flip only writes the column holding its own face's
  halfedge, and the other column can never equal it.
- **The duplicate-edge guard reads a hashed key set** (`grouping.hash_find_or_insert` plus a new
  `key_set_remove` tombstone) once the sort is stale, and the sorted keys before
  (`kernel_remesh.edge_key_exists`, a warp-uniform selector on the mask). Inserts claim only empty
  slots, so an insert racing a remove in one launch is safe; tombstones only accumulate, and
  `_FlipTopology.flipped` rebuilds once the flips since the last build could fill the table past
  three quarters. `valence` rides the commit too (one edge moves from `a`, `c` to `b`, `d`).

**The first cut regressed the short calls, and the census found why at every step.** Built with
the state eager -- set seeded, maps written, a refresh every round -- `subdivide_region_to_size`
read 0.86-0.93x: its flip calls run one round that flips little, so the machinery was pure cost
(+3 launches, +11 allocations, +1.2 ms of device time at `sphere_large` from zeroing and filling a
2n-slot table). Making the key set lazy took it to 0.94-0.96x; refreshing an issued round only when
it flipped and passing `None` / the sort buffer as the inactive state took the launch count level;
and **starting plain and taking the incremental build as the regroup after the first flipping
round** removed the rest, so a call that flips at most once costs exactly what it did. The general
form: **an incremental structure that replaces a per-round rebuild should be built by the rebuild
it replaces, the first time one is needed** -- not up front, where the calls that never iterate pay
for it. `edges_unique` re-sorts only (the rows, maps and set are current), rather than rebuilding.

A flip loop on `icosphere(4)` jittered by 0.3 of an edge oscillates at one flip a round to its
iteration cap, in the baseline too (`[354, 163, 55, 16, 6, 1, 1, ...]` over 50 rounds) -- a
pre-existing cycle, not this change; `test_flip_topology_incremental_state_matches_a_fresh_build`
uses `icosphere(3)` at the same jitter (79, 28, 9, 1 flips past the plain round), which is the
smallest fixture found that exercises the incremental rounds at all -- at a jitter of 0.02 every
flip happens in the plain round and the test passed a no-op `refresh`.

### 16.25 The round-19/20 kernel de-duplication pass (2026-09-26)

Every kernel changed in `e34051f..158af32` read against the rest of `kernels/`, by seven reviewers
with disjoint file ownership (counts as evidence, CPU byte-identity against a detached `158af32`
worktree), then a harness A/B, one pytest process per module, min of 2 rounds, triwarp rows only
(`plans/benchmark-round-21-data/ab23_*.txt`): **923 cells, median 1.01x, geometric mean 1.06x,
32 cells at 1.5x or better.** Every cell under 0.93x was re-run alone; the ones that stayed
there had identical call counts in both arms, except the three traced below, all fixed. CPU
byte-identity held over every probed output (~20 000) except `polyline_radius(reduction="mean")`,
9.4e-08 relative (summation order).

Largest wins: `polyline_point_distance` / `distance_to_closed_polyline` 24x at 4 096 queries over
65 536 segments, `bridge_edges` 21x and `bridge_edges_smooth` 13x on `lucy`, `edges_length` 5.7x
on `lucy`, `mean_edge_length` 2.4x, `crease_edges` 1.9-2.2x, `face_adjacency*` 1.6x on the large
meshes, `edge_manifold_mask` 1.8-1.9x. What generalises:

- **A starved brute-force grid whose reduction is a min or max slices its inner loop over a
  second grid dimension and commits with `atomic_min`** -- order-free, so byte-identical (§2.3's
  occupancy rule without a rewrite; `polyline.distance_to_segment_slices`). **The lanes must be the
  outer items**: `dim=(slice, query)` broadcasts each segment load across the warp, and
  `(query, slice)` measured 2-6x slower at large slice counts.
- **Validating a handful of vertex pairs needs no edge table.** One face-parallel census of each
  queried pair's undirected and directed matches answers "on the rim, wound as its face" (both
  counts 1) and "edge exists" (`holes.bridge_edge_census`); the boundary-edge build it replaced
  was 11x on `lucy`. `bridge_edges(boundary_edges=)` went with it -- nothing needed it any more.
- **The run of equal sorted keys answers "is this edge manifold"**: 1, 2 or 3+ is decidable from
  a sorted position's neighbours, so a per-face or per-halfedge manifold mask is one kernel after
  the sort, not `unique_1d` plus inverse plus counts plus a gather (1.8-1.9x).
- **§16.23's union-find edges formed in the thread generalise** -- to the vertex-manifold corner
  graph and to the orientation parity graph, byte-identical, device time flat or lower. libigl's
  "every vertex below `max(faces)` is referenced" half then runs in corner space: vertex `v - 1`'s
  corner minimum still at the `int32` seed.
- **A union-find flatten must never write into `parents`.** Writing labels over `parents` to save
  an allocation was byte-identical on CPU and wrong on CUDA (1-140 entries per mesh): another
  thread's path halving writes `parents[child] = grandparent` from a stale read, over a root
  already stored. A serial device cannot show it.
- **The two-mesh broad phase is the self-intersection one.** `intersection.collect_face_box_candidates`
  (moved from `validation`) serves both, with `candidate_slot_query` as the shared prologue. The
  fixed-stride table removes the count pass, the scan and the readback, and pays where the
  consumer is cheap (a verdict or a mask: 1.16-1.24x); where it is heavy `float64` work the
  sparse slots roughly double that kernel, so the segment compaction launches over the kept count
  (`wp.lower_bound` on the scan) instead of over the slots. Storing per-slot segments to skip the
  recompute was 0.8x: a 26 MB `wp.empty` per call cost far more than §13.1's flat 6 us (unmeasured
  why; the cold-pool reading of §13.1 is a guess).
- **The narrow fold's width is rows per block when the fold commits `float64` to constant slots.**
  Fusing a `(count, total)` fold into `points.neighbor_distance_moments` at one row per lane was
  1.4x the unfused pair at 1 M rows (15 625 blocks contending on two slots); 1 024 rows starved the
  grid at 41 k; 256 won at both (§13.2's narrow-fold rule, with its limit).
- **`wps.bsr_zeros` then field assignment wastes three allocations and a memset per matrix**;
  `wps.bsr_matrix_t(dtype)()` plus field assignment and `notify_nnz_changed` works on both devices
  (`linalg._bsr_over`). And a fresh squared-Laplacian operator of a seen shape now takes a pooled
  solver state, as §16.19's scalar solves do (`smooth_region[bunny]` 67 -> 21 launches, 1.14x).
- **A mask-then-scan compaction is deterministic where an atomic cursor is not**: `boundary.ears`
  now returns ascending face order on CUDA too.
- **Turning a loop-bound `if` into `wp.where` can move CPU results** by ~2e-7
  (`accumulate_radius_frame`): the CPU oracle is sensitive to control-flow shape, not only to
  arithmetic.

**Three regressions the integration introduced, each found by re-running the slow cell alone:**

- **A shared helper that returns trimmed views is not free to callers that pass `n`.**
  `adjacency.sorted_face_edge_keys` now writes keys and payload in one launch (1.3x), but routing
  the three internal sorters through it added ~14 us of view construction; they launch the shared
  `adjacency.face_edge_keys_and_order` kernel themselves and keep the double-width buffers.
- **A census that rides in a counting launch is paid by the inputs where the count is zero.**
  `boundary_loops_batched`'s seam/pinch census cost a closed mesh (which the baseline returned from
  early) two zeroed allocations and a 26 us slice readback: 0.83x. One zeroed buffer holding the
  flags, the two bits and the degree table, read through `read_values`, made it 1.02x.
- **REFUTED -- the flip loop under `wp.capture_while`** (an `end_flip_round` kernel keeping the
  running total and the tombstone budget on the device, one state read per entry). CUDA flips were
  identical and readbacks fell (102 -> 4 on a 100-round call), but a conditional round costs more
  than a host-replayed one and recording it costs a couple of dozen per-round reads:
  `delaunay_triangulation` 0.86-0.97x at 2 000 and 20 000 points (many flips per round, so the
  tombstone budget keeps sending it back to the host), `remove_t_vertices[saddle_graded]` 0.86x at
  a switch after 4 replays, and only a 100-round, few-flips call gained (1.08-1.13x). Reverted;
  §16.24's host-replayed rounds stand. The per-round read is not the lever there.

Also landed: the flip claim tables are re-armed by `refresh_flip_rows` (a recorded round 7 -> 4
nodes, ~1.02x -- the round is device-bound); the collapse pass seeds its claim/remap/positions in
`collapse_candidates`; `voxel_down_sample` max 5 -> 4 launches and 8 -> 5 allocations;
`statistical_outlier_mask` 4 -> 3 launches; `crease_edges` 7 -> 3 launches and 2 -> 1 readbacks;
`face_adjacency` compacts straight off `sorted_face_edge_keys`; `is_orientable` 1.17-1.47x;
`remove_duplicated_vertices` drops a readback; both ICPs keep their costs in the moments
accumulator (1-3 allocations fewer per call). Declines are at their sites.

**Fixed after the pass:** `icp_point_to_plane`'s MAD robust scale read the target normal table raw
(`robust_residual_keys`) while the fit normalized it, so non-unit `target_normals` put the scale
on residuals scaled by each normal's length. Both now read through one
`kernels/registration.target_unit_normal`, and the docstring states that normals need not be unit.
`test_robust_scale_ignores_the_length_of_the_target_normals` (same double under per-normal
rescaling) and `test_icp_point_to_plane_accepts_non_unit_target_normals` fail on the raw read.

### 16.26 Round 23: round 22's items (2026-09-27)

`plans/benchmark-round-11.md` §22.2's items, built by seven agents with disjoint file ownership
against a detached `4aa09c8` worktree (`/tmp/tw23base`), evidence counts plus CPU byte-identity;
probes are `plans/benchmark-round-23-data/probes/r23<agent>_*`. Integrated tree: full CUDA suite
green. What generalises:

- **An unbounded `max_t` / `max_dist` is bit-identical to the scene diagonal, and computing the
  diagonal cost two launches and a readback (R22-3).** `mesh_query_ray` / `_anyhit` start
  `min_t = max_t` and prune against the running best; `mesh_query_point_*` square the limit
  (`inf^2 = inf`) and only prune with it. 330 of 330 outputs identical on both devices, grazing rays
  included; `intersects_*` / `longest_ray` 1.3-1.7x, ICP on a mesh target 1.1x (it also loses the
  seed transform launch). **Three kinds of limit stay**: one that also sizes something else
  (`visibility`'s ray-origin offset, `closest_point_on_edges`' initial search radius); one inside an
  iteration whose state can outgrow it (`max_tangent_sphere`'s shrink query, where a miss *is* the
  answer -- inf changed 68-296 rows of 2 048); and one that prunes queries lying outside the
  geometry (`containing_faces_2d`: identical answer, 1.1-1.2x slower at 1 M CUDA queries and at
  every size on CPU).
- **A caller-supplied `edges_sorted` is never cheaper than packing keys from `faces`** since
  `sorted_face_edge_keys` (one launch into the sort buffer, `end_bit`) -- the rows path is a hash
  launch, a staging copy and a full-width sort. Every validation / adjacency / `Trimesh` property
  now ignores the table (still accepted and device-checked, documented "Not read"). A `Trimesh`
  property that built `edges_sorted` cold to feed a callee was pure overhead:
  `is_winding_consistent` 2.3x, `is_edge_manifold` 1.6x, `face_adjacency` 1.3-1.5x, cold
  `is_watertight` 1.69x open / 1.10x closed -- the last also because `is_watertight(mesh=)` now
  takes a factory and builds the BVH only once both manifold tests pass. **Declined: the early exit
  on the edge test** -- reading the edge flag before issuing the vertex test saves an open mesh a
  third and costs a closed one 3-8 %, because the vertex test's launches stop overlapping the sort.
- **`remove_degree3_vertices` validates inside pass 0** (the halfedge-twin rule counted in the fan
  launch, carried in its readback): 1.12-1.40x. The rule itself is one `@wp.func`,
  `kernels/halfedge.sorted_halfedge_run_class`, which `pair_sorted_halfedges` also calls -- it
  cannot live in `kernels/adjacency.py`, which imports `halfedge`.
- **Boundary walks rank their cycles with `_closed_successor_cycles`, not the general
  `successor_cycles` (R22-2).** A window table `(succ^W, window minimum, hops to its first
  occurrence)` merged with a strict `<` gives every node its cycle's start and its distance in one
  set of rounds; one `vec2i` scan places every cycle. The dart walk's mirror filter and the pinched
  walk's gather moved onto the device, and the pinched walk builds its twins off
  `_BoundaryHalfedges`' own sort. `boundary_loops_batched` 2.2-2.8x on the rim fixtures, 1.7x
  `dragon`, 1.3x `lucy`; `delete_region_keep_boundary` 1.8x; `fill_fan[holes_many]` 1.7x. **The
  plan's premise "pure cycles by construction" is false for the vertex walk on a mesh that is not
  edge-manifold** (`bunny_decimated`: 71 of 123 boundary edges lie in loops), so the successor table
  is `-1`-filled and a chase that reaches a non-tail drops the node -- an uninitialised table
  segfaulted the CPU device mid-edit. **Pointer jumping chases up to `POINTER_JUMP_MAX_HOPS = 16`
  pointers a launch, and the clock is flat from 8 to 64 hops** (`rim_long` 0.633 / 0.512 / 0.463 /
  0.454 / 0.451 / 0.443 ms at 2 / 4 / 8 / 16 / 32 / 64): most of the win was dropping the general
  pipeline, not the width. **In NumPy multi-hop loses** (`h - 1` gathers a round: 1.3-2.7x slower
  than doubling), so `intersection`'s host doubling keeps doubling.
- **`marching_triangles` links on the device from `_LINK_ON_DEVICE_FROM = 1024` segments, CUDA only
  (R22-6).** The compaction writes the `2n` crossing keys straight into a radix sort (`end_bit`);
  adjacent equal keys pair segments (0.08-0.11 ms flat against `np.unique`'s 0.11-1.14 ms at
  1.5 k-26 k); multi-hop ranking carries `(ahead, lowest, first-min offset, steps)`; one weight scan
  lays the curves out in the host's order; a malformed pairing falls back to the host link, keeping
  its `ValueError`. 1.27x at 1.5 k segments, 1.5x at 26 k, 2.2x at 53 k; on the CPU device it loses
  at every size (0.89-1.0x). `split_faces_along_field` sorts only the cut faces' crossed halfedges
  instead of a whole-mesh `edges_unique`: 1.5-3.1x. **`edges_unique`'s row order is `(max, min)`
  lexicographic** (`hash_indices_rows` packs column 0 as the low digit), so a region sort meant to
  reproduce its numbering keys `max * base + min`.
- **The k-NN hash grid's deferral** is §16.6's rewritten bullet (R22-4).
- **Small trims (R22-7 and the passes around them).** `crop_points` / `points_in_*` flag, scan in
  place, compact: 1.5-1.7x. `cluster_decimate` packs its cell keys in the cell kernel: 1.05x.
  `unique_1d`'s hash occupancy is scanned in place and counts are allocated only when asked (one or
  two table-sized allocations fewer per call; `unique_faces` 1.14x). `outlier_probability` forms
  its normalizer on the device (1.34-1.41x) -- and **`wp.utils.array_inner(a, a)` returns an
  `np.float32`**, so the host's `value / n` was a float32 division under NEP 50 and the device
  port must reproduce it or sit 1 ulp off. `normals_at_closest_faces` without a table forms the hit
  face's normal in the query thread. `intersects_location` returns prefix views of its per-ray
  buffers. `geodesic_ball` reads its overflow counter with its total. **A box predicate with an
  optional rotation must branch, not multiply by the identity**: `0 * inf` is `nan`. The two face
  compactions share `kernels/triangles.copy_scanned_face`. `remove_degenerate_faces`' fusion was
  declined (at most one launch, <= 1.07x).
- **Agent hygiene, two more.** `ruff format triwarp/` run by one agent reformats files others are
  editing mid-Edit: always name your files. And `pkill -f <probe>` kills the shell running it, the
  same self-match as §11's `pgrep`.
- **R22-9: one offsets convention, and a packed sibling for every list return.** §4.2 has the
  rule. Built in two passes: the convention and the `_batched` renames (`split_with_offsets`,
  `boundary_loops_with_offsets`, `loop_perimeters_from_offsets`, `loop_directed_areas_from_offsets`;
  `include_total=` removed everywhere, and `proximity.query_mesh_aabb_with_offsets` /
  `graph.successor_cycles` / `selection.submeshes_from_face_groups` lost their sizes arrays) —
  byte-identical on CPU over 697 outputs, parity unchanged at 585 pairs — then
  `marching_triangles_with_offsets`, `delete_region_keep_boundary_with_offsets` (the classifier writes `(kept, size,
  start)` per loop; the packed form scans it in place and compacts on the device, the list form
  reads it raw and cuts its views from the traced loops -- one readback either way, and the list
  form pays no scan),
  `homology_generators_with_offsets` and `shorten_loop_with_offsets`, each list form rebuilt on
  it, 607 list outputs byte-identical on CPU. `repair.remove_tunnels` now stays packed from the
  basis to the measurement, where it went list -> packed -> list -> packed. **`query_ball`'s list
  form cloned every query's segment** (8 195 allocations at 4 096 queries): views through
  `array.split` made it 6.7x (68 -> 10 ms), and a `copy=` keyword keeps the old behaviour
  reachable.
  **A default result built eagerly is two allocations on every call.** The first cut built
  `delete_region_keep_boundary`'s empty `(values, [0])` return up front for its four early exits
  and read 0.91x on the harness; building it only when returned put it back at parity. An
  empty-return tuple in a wrapper is a function, not a value.

### 16.27 Round 24: the sweep, and allocating less instead of keeping the pool (2026-09-28)

Full sweep of `4aa09c8` plus the round-22/23 working tree (`plans/benchmark-round-24-data/`): gap
60.9 -> **40.0 ms**, 767 wins of 928, 5 slower cells of which 4 re-measure level and one is real --
`interpolate_from_points[lucy k=8]` 0.66x, because its radius path now defaults to the hash grid
and `hashgrid_from_points`' 128^3 bins are far fewer than `lucy`'s occupied cells (plan R24-1).

- **A drained mempool is a size-proportional cost, and it is the host time of few-launch rows on
  large meshes.** `crease_edges[lucy]` read 19.0 ms wall against 2.5 ms of device time with three
  launches, `cotmatrix[dragon]` 2.95 against 0.41. Raising the release threshold was 1.5-1.6x on
  both at `dragon` / `lucy` and flat at `bunny` (`probes/r24_mempool.txt`). triwarp does not set
  it -- the setting is process-wide -- so the lever is fewer and smaller allocations. **Census
  allocation *sizes*, not counts** (`alloc_plugin.py`, which patches
  `CudaMempoolAllocator.allocate` and attributes each allocation to its triwarp frame): a count
  census prices the host calls and cannot see this.
- **`cotmatrix` allocated 9.4 GB per call at `lucy`** for a matrix of under 1 GB: three
  12-triplets-per-face buffers of 1.28 GB each and 5.2 GB of `bsr_from_triplets` scratch. It now
  emits the six off-diagonal triplets per face plus one diagonal slot per vertex (the tail's rows
  prefilled with `n_vertices`, so an unreferenced vertex's slot is out of range and dropped and its
  row stays empty), and `kernels/laplacian.cotmatrix_diagonal` writes each diagonal as minus its
  row's off-diagonal sum. **2.00x at `dragon`, 1.94x `happy_buddha`, 1.79x `lucy`** (102.5 -> 57.1
  ms), 1.25x `bunny`, **0.93x `bunny_decimated`** (the extra fill and launch, ~24 us). Pattern and
  off-diagonals bit-identical on CPU (an off-diagonal's terms and their order are unchanged), the
  `float64` diagonal bit-identical, the `float32` diagonal within 3.2e-7 relative -- its terms are
  now summed per edge first. The capacity `nnz` overshoots `nnz_sync()` by 1.85x rather than 3.4x,
  and the duplicate-built-operator tests still reach a duplicate build. After it the pool still
  costs 1.5-1.7x of what is left (`probes/r24_mempool_after_cotmatrix.txt`): that was
  `bsr_from_triplets`' own sort, removed next.
- **Not transferable to `connection_laplacian`**: its diagonal is `sum w * I` and `w` is not stably
  recoverable from a rotated off-diagonal block (the decode divides by `cos rho`), so it needs a
  scalar weight source, and a float64 atomic sum would make the diagonal nondeterministic on CUDA.
- **Then every `bsr_from_triplets` call went (§3.7, check 27).** The sorted-keys core
  (`array.csr_from_keys`: one `uint64` key per contribution, a radix sort over only the bits
  `n_rows * n_cols` needs, run starts scanned in the payload buffer's own upper half, offsets
  filled by each row's first run so empty rows need no pass) replaces it at all sixteen sites --
  structurally for the mesh operators, through `csr_from_triplets` elsewhere, directly for the
  prolongator and the empty matrix. Harness A/B against `4aa09c8`, one process per module
  (`plans/benchmark-round-24-data/ab_sparse_*.txt`): `cotmatrix` **2.90x `dragon`, 2.62x
  `happy_buddha`, 2.44x `lucy`**, 1.62x `bunny`, 1.19x `bunny_decimated` (the round's small-mesh
  loss gone); `robust_laplacian` 1.52-1.64x; `connection_laplacian` 1.13-2.01x;
  `laplacian(equal_weight)` / inverse-distance 1.07-1.47x; `laplacian_smoothing_loss[cotcurv]` up
  to 2.03x; the energies 1.05-1.28x; `graph`, `subdivide_loop`, heat, parametrization, multigrid and
  the implicit smoothers flat to 1.14x -- the solve dominates there. Every sub-0.95x cell re-measured
  level alone. `csr_from_triplets` is bit-identical to `bsr_from_triplets` on 160 random scalar
  builds on both devices (duplicates, out-of-range, empty rows, pruning); the mesh operators'
  off-diagonals and patterns are bit-identical, their diagonals now a row sum.
- **Warp's `prune_numerical_zeros` prunes triplets, not entries**: a zero-valued triplet is dropped
  before summing, and an entry whose nonzero triplets cancel is kept holding zero.
  `csr_from_triplets` prunes the assembled entries instead (the more useful answer), sums each run
  at its start in the same launch that flags it (no separate sum kernel, no `starts` buffer), and
  still skips zero triplets before the sort. Priced against Warp at 0.2 M / 4 M / 40 M triplets:
  1.06x / 1.00x / 1.10x pruned, 1.12x / 0.99x / 1.11x unpruned -- a wash with the pre-sum rule
  (`probes/r24_csr_prune.txt`), so the semantics were chosen for the answer, not the clock.
  Correctness gate (`probes/r24_csr_check.py`): 360 random builds per device, every entry we keep
  equals Warp's, every entry Warp keeps that we drop is zero there -- exactly on CUDA, at most
  9.5e-7 on CPU, whose own summation order is not the triplets'. The generic builder is only at
  parity with Warp's (0.96-1.13x); **the big wins are the structural builds**.
- **The undirected mesh-pattern build (above ~0.5 M faces)**: one sort of `3 * n_faces` keys plus
  a 32-bit sort of the unique edges, instead of one sort of `6 * n_faces + n_vertices`. The sorts
  alone are 1.70-1.88x at `dragon` / `lucy` and 0.72x at the bunnies (`probes/r24_sort_split.txt`);
  the whole `cotmatrix` 1.35-1.45x above 0.87 M faces and 0.55-0.78x below 0.33 M
  (`probes/r24_pattern_crossover.txt`). Two things decided it: the second sort's 32-bit keys live
  in the first sort's upper-half key scratch (a `wp.array(ptr=...)` alias), and each entry's run
  end is read off the sorted keys rather than stored -- together `lucy` went 6.6 -> 5.3 GB and the
  first version's 1.11x became 1.35x. A degenerate face's self-edge, which the directed build
  keeps as its keys, is carried as a flagged diagonal slot so both builds agree.
- **`graph_laplacian` and `laplacian` (given faces) went structural too.** `graph_laplacian` had
  run `edges_unique`, a triplet build, `bsr_mv`, `bsr_diag` and `bsr_axpy`; it is now one pattern
  and one row kernel -- **3.0-5.2x**, `lucy` 101 -> 29 ms. `laplacian_equal_weight` (directed, one
  key per halfedge) 1.42-1.68x; the symmetric branch 1.64-2.52x against the triplet path in one
  process (`probes/r24_laplacian_structural.txt`). All 50 operator matrices (5 meshes, degenerate
  and open included, 2 precisions, 5 operators) bit-identical to `4aa09c8` on both devices.
  Harness, final: `cotmatrix` 3.1-3.95x at `dragon` / `happy_buddha` / `lucy`, 1.16-1.51x at the
  bunnies; `connection_laplacian` 1.15-1.83x; `robust_laplacian` 1.7x. The
  `laplacian_inverse_distance` row passes precomputed `edges=`, so it measures `csr_from_triplets`
  against Warp (0.97-1.13x), not the structural build.
- **The sites left on `csr_from_triplets`, and why**: the energy sandwich products and hessians
  (each input row scatters over many output rows -- unordered by construction), the Loop operator
  (its even rows are unsorted rings, and a per-row sort is §16.6's hub hazard), `edges_to_csr` /
  `index_sparse` (the input *is* a coordinate list), the implicit smoothing system (solve-bound,
  0.97-1.04x whole-call).
- **Gates at the end of the round**: `tests.devices` CUDA 3 857 passed / CPU 3 811 passed,
  `tests.parity` 585 pairs / 30 exempt / 0 uncovered (unchanged), basedpyright 0, ruff clean.
- **The rest of the census is functional**: `crease_edges`' 1.9 GB at `lucy` is the radix sort's
  double-width keys and payload (the key needs 48 bits there), `cluster_decimate`'s hash tables and
  face buffer, `mesh_to_mesh_distance`'s 963 MB of per-face and per-vertex buffers.

