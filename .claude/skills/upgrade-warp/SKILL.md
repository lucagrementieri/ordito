---
name: upgrade-warp
description: Upgrade ordito's warp-lang dependency to a new NVIDIA Warp release, or evaluate one before upgrading. Covers surveying the release page, changelog, milestone and commit range; bumping the lock; repairing private-API breakage; auditing semantic changes and fixes ordito bypasses; regenerating the API mirrors; re-probing version claims; running the gates; pricing performance against the old version; and writing the adoption plan. Use whenever the user asks to upgrade, bump or move to a new Warp version, or asks what a Warp release means for ordito.
---

# Upgrading the Warp dependency

An upgrade is two deliverables:

1. **The upgrade itself**: a commit on `main` with the lock bumped, every breakage repaired, every
   stale claim re-verified and the gates green.
2. **An adoption plan** (`plans/warp_<version>_upgrade.md`, which is gitignored): what the release
   makes possible, slower or wrong, each item with a measurement or the words "needs measuring",
   ranked, each with a decision rule.

`.claude/CLAUDE.md` is the rulebook throughout. The sections an upgrade leans on are §9
(measurement method), §10 and `reference/warp_api/REGENERATE.md` (mirrors), §4.5 check 9 (version
claims), §12 (platform facts, the §12.10 workaround table), §13 (cost model), §15 (measurement
traps) and §8 (commit and gates). Read §12.10 before starting: it is the list of things the last
upgrades re-verified.

Throughout, `OLD` and `NEW` are the version strings (e.g. `1.17.0` / `1.18.0`).

---

## 0. Preflight

- **Clean tree, note the baseline commit.** The pre-upgrade `HEAD` is the A/B baseline for the
  rest of the job. Create a detached worktree of it now (§15.6; never `git stash`):
  `git worktree add -q --detach $SCRATCH/base HEAD`.
- **Driver vs toolkit.** Read the release's platform requirements (CUDA toolkit the wheels are
  built with, minimum driver, minimum GPU arch). Compare them with `nvidia-smi` and with what
  `wp.init()` prints after the bump. A driver older than the wheel's toolkit can still run under
  minor-version compatibility; the proof is one real launch, not the version numbers.
- **The old Warp stays runnable without touching `.venv`**:
  `uv run --no-project --with warp-lang==$OLD --with numpy --with trimesh python <probe> <tree>`.
  A probe importing ordito must drop the editable MetaPathFinder, put the tree first on `sys.path`,
  and assert the *submodule's* `__file__` (§15.6). Use the baseline worktree as the tree for the
  old arm.

## 1. Survey the release

Run `scripts/release_survey.sh OLD NEW $SCRATCH/survey`. It writes:

- the `## [NEW]` section of the changelog;
- the GitHub release body;
- every closed issue and PR in the milestone named for the release;
- the full commit range `vOLD..vNEW`;
- the commit range **filtered to the Warp files ordito depends on**;
- a blobless clone, for `git -C warp-src diff vOLD vNEW -- <file>` on demand.

Read all of it. The changelog is the summary. The milestone holds bug reports whose titles say
what was broken, sometimes more plainly than the changelog entry. The path-filtered commits tell
you **which of ordito's dependencies actually moved**: a file with zero commits in the range is a
free control row for later timing.

**Ordito's dependency surface** (extend the script's `WATCHED` list when the tree grows a new one):

| Warp file | What ordito rests on |
|---|---|
| `_src/context.py`, `_src/codegen.py`, `_src/types.py` | `ordito/_launch.py`'s cached launcher and allocators drive these privately: launch bounds, `Adjoint` / `Module` attributes, `pack_arg`, the native launch prototype |
| `_src/utils.py` | wrappers (`array_scan`, `radix_sort_pairs`, `array_cast`, `segmented_sort_pairs`) whose native calls `_launch.py` makes directly, bypassing the wrapper's checks |
| `_src/sparse.py` | §3.7's `nnz` semantics, the `odt.bsr_*` typed views, the multigrid fallback |
| `native/bvh.h`, `mesh.h`, `hashgrid.h` | every neighbour, proximity, ray and curvature query; their relative prices set backend defaults and leaf sizes |
| `native/tile*.h` | tile reductions, `launch_tiled` lane behaviour (§12.2), the block-barrier tricks (§14.11) |
| `native/builtin.h`, `range.h` | integer arithmetic semantics (§1.5), the empty-range adjoint (§12.4) |
| `native/sort.cu`, `scan.cu` | radix sort `end_bit`, scan status and capture rules |
| `_src/fem`, `_src/optim/linear.py`, geometry / marching-cubes modules | the adaptive Poisson path, CG's capture behaviour (§12.7), level-set extraction |

**Triage every item** of the changelog and the milestone into exactly one bucket. Grep ordito for
each name; do not assume. The bucket decides which later step handles it:

| Bucket | Handled in |
|---|---|
| **Breaks us**: removed or renamed public API, a breaking change, a private attribute `_launch.py` reads | step 3 |
| **Changes semantics** we rely on: arithmetic, rounding, defaults, return types, `nnz`-like counters, exception types | step 4 |
| **A fix ordito bypasses**: the fix sits in a Python wrapper whose native ordito calls directly | step 5 |
| **Moves performance** of something we use: a rewritten traversal, a faster dispatch, a new compiler path | step 9 |
| **New capability** with a plausible ordito consumer | step 10 |
| **Irrelevant**: JAX, textures, rendering, platforms we do not ship. Say so in one line of the plan, so nobody re-triages it | — |

## 2. Bump

```bash
uv lock --upgrade-package warp-lang
uv sync --all-groups          # a bare `uv sync` uninstalls the test dependencies (§8)
uv run python -c "import warp as wp; wp.init()"   # toolkit, driver, devices, cache dir
```

Then launch one tiny kernel on **each** device and read the result. The kernel cache is namespaced
by version, so the first suite run recompiles every module cold. Expect long stretches at 0 % GPU
(§15.1); it is not a hang.

## 3. Make it run

- **Run the CUDA suite with `-x -W default::DeprecationWarning`** in the background. The first
  failure is usually in `_launch.py`, because it touches the most private surface.
- **Private internals.** For every `warp._src` name `ordito/_launch.py`, `_device.py` and any other
  module touch (`grep -rn "_src\.\|_ctx\.\|_types\.\|_codegen\." ordito`):
  1. Confirm it still exists.
  2. **Diff the Warp function that computes it** between the two tags. A name can survive with a
     new meaning: a bound that covered one axis now covers all of them; a return that was `None`
     is now a status.
  3. Diff the native prototypes (`runtime.core.<fn>.argtypes` / `.restype`) and the launch-bounds
     struct layout.

  Mirror the new semantics conservatively: when unsure, defer to the public `wp.*` call, which is
  `_launch.py`'s documented fallback.
- **Removed and deprecated names**: grep `ordito/`, `tests/` and `benchmarks/` for each one in the
  changelog. Migrate deprecated APIs now, while the old spelling still works. Update the prose that
  names them as well (comments, docstrings, CLAUDE.md), reflowing to 100 columns.
- **Type stubs**: run `uv run basedpyright`. A release that tightens Warp's stubs produces new
  errors at call sites that were fine. Fix them in §8's preference order: an `odt` helper first,
  then a commented `cast`. Also check the opposite direction: `reportUnnecessaryCast` and
  `reportUnnecessaryTypeIgnoreComment` flag what the new stubs made redundant.

## 4. Audit semantic changes

For each "changes semantics" item:

1. **Probe it** with a few-line kernel (a file; Warp refuses `exec`-defined kernels). Run it on CPU
   and CUDA, under OLD and NEW, and record the exact outputs. Test the neighbouring operations too:
   a release can change one operator and leave its partner alone (e.g. a division whose remainder
   did not follow), which breaks identities that held before.
2. **Enumerate every ordito site** that uses it (`grep`, AST scan). Classify each one: provably
   unaffected (state the invariant: a guard, a non-negative domain, host scope), affected and
   fixed, or affected and intended.
3. **Price it if it adds work**: count SASS per kernel entry from Warp's cache (§12.6's recipe; note
   whether the new version caches PTX or cubin). A few integer instructions per site is normally
   noise; a hot loop is not.
4. **Update the rule** in CLAUDE.md Part I if the change invalidates one, and the Part II fact with
   the numbers.

## 5. Port fixes ordito bypasses

`_launch.py` exists to skip Warp's per-call Python overhead, so it also skips whatever checks a
release adds to those Python wrappers. For every changelog or milestone fix located in a wrapper
whose native ordito calls directly:

1. Diff the wrapper between tags.
2. Port the new check (a status check, a bound, a raise) into the fast path, with a comment naming
   the issue.
3. Run the tests that exercise that path, including those that record it into a CUDA graph.

A fix in a native function (`.h` / `.cu`) reaches ordito automatically. A fix in Python may not.

## 6. Regenerate the API mirrors

Follow `reference/warp_api/REGENERATE.md` exactly. It is mechanical, so delegate it to a
background subagent while the suite runs, and pass the changelog path along.

- Transcribe from the docs version that matches the install (`stable` vs `latest`; check the
  page's version stamp).
- Add a mirror file for any new public module.
- Run the argument-order check; it must print nothing.
- `reference/` is gitignored: the mirrors are local and never appear in the commit.

## 7. Re-verify the version claims (check 9)

`tests/test_api_conventions.py::test_warp_version_claims_are_not_stale` fails on every comment
naming an older `Warp 1.N`. Each site is exactly one of three kinds:

- **Mechanism claim**: re-probe it on NEW (throwaway subprocesses for anything that can corrupt
  the process; repeat a flaky failure ~10 times). Re-stamp it to NEW if it still holds, or fix the
  text if it does not.
- **Measurement stamp** (a ratio or table taken on OLD): do not re-stamp without re-measuring.
  Allowlist it, and list it in the plan as a re-measurement owed.
- **Genuine history** (the version an API appeared or a behaviour changed): allowlist it.

Allowlist entries go into a new block headed `# Added by the Warp <NEW> upgrade.`, whose preamble
lists the mechanisms re-probed and their results (the precedent is the existing blocks in
`_WARP_VERSION_ALLOWLIST`). This is also mechanical: delegate it to a background subagent, with
instructions to report every claim that **changed** and every comment found wrong.

Correct the wrong comments, even if they were already wrong on OLD.

## 8. Floor and documentation

- **Raise `warp-lang>=` in `pyproject.toml`** only if the tree now uses an API OLD lacks (an
  import of a new module counts). Otherwise keep the floor (§12.10).
- **Update every user-facing mention** of the minimum version (`README.md`, `docs/index.md`,
  `docs/getting-started.md`), plus any new driver or GPU requirement from the release.
- **Update CLAUDE.md**:
  - the "Installed:" stamp in the preamble;
  - the §12.10 workaround table (a row per re-verified workaround, a row per new hazard);
  - each Part II section a finding belongs to, next to the numbers it revises.

  Then run check 24: `pytest tests/test_api_conventions.py -k claude`.

## 9. Gates

All of them, on the final tree:

```bash
uv run python -m tests.devices                       # CUDA pass, then CPU pass in a fresh process
uv run basedpyright                                  # 0 errors
uv run zensical build --strict --clean               # "No issues found", and site/ has pages (§6 inotify trap)
uv run python -m tests.parity                        # pair count == the baseline worktree's
uv run ruff format --check ordito tests benchmarks && uv run ruff check ordito tests benchmarks
```

If any code changed after a gate ran, re-run at least the tests covering that code, and say so in
the hand-over.

## 10. Price performance against OLD (quiet box only)

Clock readings are invalid while pytest, a docs build or another probe runs (§11), so this step
comes after the gates. §9 and §15 govern method. The rules that matter most here:

- **Measure through ordito's public calls**, at the benchmark operating points and on ordito's
  input class: scan meshes, surface point clouds. A primitive timed on synthetic uniform data can
  move in a different direction, and by a different amount, than on real inputs.
- **Include the structure build where the default call pays it.** A default is decided for the
  call without a prebuilt accelerator.
- **Carry a control row** the release did not touch (step 1's zero-commit files). It licenses the
  comparison.
- **Alternate OLD and NEW** in back-to-back processes and repeat. Report the min and the median.
- **When a primitive moved, re-decide everything that rests on relative prices**: backend
  defaults, query shapes (box vs sphere), leaf sizes, crossover thresholds, size gates. Sweep them
  across the **size axis**, not one mesh: an optimum that moves with size wants a gate, and one
  that does not wants a single default. Choose a rule by worst-case regret against the best
  setting per cell, and count the cells from the data, not by eye.
- **A regression in Warp gets an upstream issue with a minimal repro** before any workaround is
  built.
- **Re-probe tuning constants** (§9), and regenerate `docs/benchmarks.md` once the plan's
  performance items land.

## 11. Evaluate new capabilities

For each "new capability" item with a plausible consumer:

1. **Identity**: does it produce the same answer as ordito's implementation on the same input
   (byte-identical, or a named transform)?
2. **Price**: time it at several sizes against ordito's path; crossovers are common.
3. **Classify** it as one of:
   - **adopt** (with a size gate if there is a crossover);
   - **test oracle** (an independent implementation with an identical answer is a Class A
     comparison, §7.4);
   - **opt-in test mode** (e.g. a debug or equivalence execution mode too slow for production);
   - **decline**, with the number.

   A capability with no in-repo consumer is declined (§4.2, no speculative generality).

## 12. Write it down and commit

**The plan** (`plans/warp_<NEW>_upgrade.md`) has these sections:

1. A header table of headline measurements, OLD vs NEW, each marked **[M]**.
2. What the upgrade commit did, plus its gate results.
3. Priority items, each with its measurement, method and decision rule.
4. Re-measurements owed: the allowlisted measurement stamps and the tuning constants.
5. One row per milestone item saying what it means for ordito.
6. What was declined, with the reason.
7. A suggested execution order.

Every number in the plan must come from a run you made; mark anything else "needs measuring".

**The commit**, on `main` (§8):
- message `:arrow_up: Upgrade to Warp <NEW>: <the repairs, comma-separated>`;
- a body listing each repair and the gate results;
- the attribution trailer;
- no `plans/`, no `reference/`, no scratch probes.

Remove the baseline worktree after the last A/B.

---

## Working pattern

- **Parallelise the mechanical work.** Mirror regeneration (step 6) and claim re-probing (step 7)
  go to background subagents while the cold-compiling suite runs. Give each agent the exact
  scope: which files it may edit, and that it must not commit or edit CLAUDE.md.
- **Probes are disposable.** Write each in the scratchpad for the question at hand, under both
  versions; do not grow a library of them. What persists is the number, written at the site it
  decides and in CLAUDE.md Part II.
- **Report outcomes faithfully.** Say which code landed after which gate ran. Correct an earlier
  headline when a better measurement contradicts it.

## Traps

- **A private attribute can be renamed without any import-time error**: the whole suite fails on
  its first launch. Read the first failure's traceback before assuming many independent breakages.
- **A fix can land in a Python wrapper ordito bypasses** (step 5). Read every "Fixed" entry against
  `_launch.py`'s list of direct native calls.
- **The first suite run after a bump is slow** because every module compiles cold; the second run
  is several times faster. Do not read the first run's wall clock as a regression.
- **`stable` docs can lag or lead the install**; `latest` is the next development version.
- **Dense new features** (an extractor, a solver, a geometry routine) often match ordito's answer
  exactly while losing at small sizes. Price several sizes before declaring a verdict.
- **zsh**: `grep --include=*.py` fails on an unmatched glob. Quote it, or grep the directory.
  `pytest -n` is not available here.
- **Never `pkill -f` or `pgrep -f` a probe by name** (§11). Wait on a backgrounded command's
  notification.
