"""
Static scan of the public API's shape: names, summaries, file layout and module boundaries.

Each check is a defect class that was actually found rather than an aesthetic preference. They run
as an ``ast`` scan of ``triwarp/`` (excluding ``kernels/``, ``__init__.py`` and private ``_*.py``
modules) plus a listing of ``tests/`` and ``benchmarks/``, and
[`tests/test_api_conventions.py`](test_api_conventions.py) fails the default test run on any
violation. The authoritative list with its reasoning is ``.claude/CLAUDE.md`` section 4.5; what
follows is the one-line claim each check makes, so a failure message reads in context.

1. **A one-line summary says what the function returns, not which C++ call it wraps.** mkdocstrings
   renders the summary as the entry in the module's API index, so a library name there turns the
   index into a table of bindings. Attribution is wanted -- one line down, in ``Notes`` or ``See
   Also``.
2. **A ``*_mask`` producer returns a boolean array.** A function that *consumes* a mask is named
   for its input and is exempt.
3. **A module summary does not end in "(Warp)" or "on NVIDIA Warp".** The whole package is Warp.
4. **Every module has a test file and a benchmark file named for it**, and every ``test_*.py`` in
   either suite corresponds to a module. The package is flat, so the mapping is the module name.
5. **A private name stays inside its module.** A ``_helper`` imported across a module boundary is a
   function that should have been public, and the alias-on-import is the tell.
6. **Two modules do not export the same public name**, outside a written allowlist.
7. **A top-level kernel module is named for the public module it backs**, and vice versa (section
   3.1). This is what stops a wrapper from being created while its kernels stay under the old name.
8. **A private helper is defined below its first caller** (section 5's stepdown rule), so a reader
   never jumps backward to a definition they have not met.
9. **A Warp-version claim names a version at least as new as the installed ``warp-lang``.** It also
   scans ``kernels/``, ``tests/`` and ``benchmarks/``, because a stale ``pytest.skip`` is worse than
   a stale comment: the comment misinforms, the skip silently deletes coverage, and on a box with
   CUDA the deleted branch is the one nobody runs. It abstains on ``.md``.
10. **A Python-scope allocation names the device it allocates on.** Without ``device=`` a buffer
    lands on Warp's *current* device rather than the device of the arrays it is about to be used
    with. The suite never catches it, because a test runs with its arrays' device current and the
    omission then resolves correctly by accident.
11. **A public function that raises documents a ``Raises`` block.** Only a *direct* ``raise`` in the
    function's own body counts; the many functions delegating validation to a shared guard are
    correct and are not scanned.
12. **A fenced ``python`` docstring example runs.** The odd member of the family: the scan only
    *extracts* the blocks and [`tests/test_api_conventions.py`](test_api_conventions.py) executes
    them against a mesh fixture, because an example's defect is a runtime one that ``ast.parse``
    cannot see.
13. **A kernel output argument is named ``out_*`` and sits at the end of the signature** (section
    2.1). Two argument classes are exempt and carried in ``_KERNEL_OUTPUT_ALLOWLIST``: in-place
    arguments, and scratch / persistent-state buffers.
14. **An array annotation is subscript-style** -- ``wp.array[T]``, not ``wp.array(dtype=T)``
    (section 1.2). Restricted to *annotation* positions, which is what lets it scan the whole
    package: ``wp.array(dtype=T)`` is a legal allocation at Python scope.
15. **A ``wp.launch`` / ``wp.launch_tiled`` names the device it launches on.** The memory-safety
    guard of the family: an omitted ``device=`` runs a kernel on ``cuda:0`` over host pointers,
    returns the right answer, and corrupts the heap when those arrays are freed mid-kernel. This is
    the half that carries the load -- ``conftest.py``'s ``STRICT`` mode only bites when the arrays
    are *not* on the launch device, so on a CUDA run the omission is invisible to it.
16. **A cast inside a kernel is spelled ``wp.int32`` / ``wp.float32``, never bare ``int`` /
    ``float``** (section 1.3). Same builtins under a different name, with one asymmetry that
    matters: ``float(...)`` is a *hard compile error* inside a ``wp.Float``-generic function, so it
    silently forecloses genericising that function.
17. **An integer division inside a kernel is spelled ``//``, never ``/``** (section 1.5). On
    integers the two are the *same* operation in Warp, so this is legibility: ``/`` on two
    ``int32``s reads as real division and truncates only because the operands happen to be integers.
    It types an operand only *by declaration*, which is what makes it safe on float-heavy code.
18. **A kernel-scope argument or return is annotated in Warp's types** (section 1.2). It reads
    ``@wp.kernel`` / ``@wp.func`` signatures only, because a kernel *factory* is ordinary Python
    whose ``int`` parameters are correct.
19. **A test comparing against a reference library says which class the comparison is** (section
    7.4). It accepts all four label phrases the suite uses, keys on ``ast.Assert`` so a fixture
    unpack is not a hit, and leaves ``_np`` out because it marks inputs as often as oracles. It
    checks that a label is *present*, never that it is the right one.
20. **A conditional value in kernel scope is ``wp.where``, not a Python ternary** (section 1.5).
21. **Nothing under ``triwarp/`` names MeshLib or promesh** -- a licensing guard (section 7.6).
22. **A single-index ``wp.tid()`` is cast** (section 1.3).
23. **A ``@wp.func`` reached by ``wp.map`` from several call sites has a declaration table**
    (section 3.5).
24. **A ``.claude/CLAUDE.md`` cross-reference names a section that exists**, and names a
    *subsection* wherever the chapter has any. The second half is the point: a bare chapter number
    still *resolves*, which is exactly what a resolving check cannot see. Chapters with no
    subsection are accepted bare, read from the heading structure rather than a list.
25. **No ``!!!`` admonition sits inside a numpydoc item-list section** (section 6). griffe reads
    each entry's first line as a name, so an admonition header between two ``Raises`` entries
    renders as an exception type.
26. **A Warp-typed module constant is not used as a Python-scope arithmetic operand or slice
    bound** (section 4.5). Its operators route through Warp's builtin dispatch.
27. **No ``warp.sparse`` triplet build under ``triwarp/``** (section 3.7). An operator's matrix
    is assembled from sorted keys (``array.csr_from_keys``) or, for genuinely unordered input,
    ``array.csr_from_triplets``.
28. **Every public module is shelved in ``docs/SUMMARY.md`` exactly once.** Zensical's
    ``api-autonav`` builds a page for each module, but the nav is authored by hand, and a module
    missing from it is built and unreachable with ``zensical build --strict`` still green.
29. **An array's entry opens with its shape as a code span** (section 6): ``(3 * n_faces,)`` flat
    triangle index buffer, not ``Length-``, ``Shape ``(``, ``Flat `` or ``Rank-1 `` first.

Why a static scan rather than importing ``triwarp``
---------------------------------------------------
Importing would make the verdict depend on Warp's module cache and on which optional dependencies
resolve, and would say nothing about files (checks 4 and 7) at all. A scan reads the tree as
written, so its answer is the same in every environment and every pytest invocation. Check 9 needs
the installed Warp version and takes it from ``importlib.metadata`` rather than
``warp.config.version``, so even that one imports nothing. Check 12 is the single exception,
deliberately: an example's defect is a *runtime* one, so nothing short of running it finds it, and
the extraction half stays here so the execution half has no parsing to do.
"""

from __future__ import annotations

import ast
import functools
import re
from collections import defaultdict
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed_version
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent
_PACKAGE_DIR = _REPO_ROOT / "triwarp"
_KERNELS_DIR = _PACKAGE_DIR / "kernels"
_BENCHMARKS_DIR = _REPO_ROOT / "benchmarks"

# --- check 1 -----------------------------------------------------------------------------------

# Reference libraries whose name in a *summary* line is the defect. Spelled as a single alternation
# so the allowlist below is the only escape, and it is per-module rather than per-name.
_LIBRARY_IN_SUMMARY = re.compile(
    r"MeshLib|MeshLab|libigl|pymeshlab|igl::|geometry-central|potpourri|VCG|promesh|trimesh\."
)

# ``mesh.Trimesh`` genuinely mirrors ``trimesh.Trimesh``, and that is load-bearing: the property
# names (``.centroid``, ``.volume``, ``.euler_characteristic``) are chosen to match, so a reader
# needs to be told at the top. No other module may name a reference library in a summary.
_SUMMARY_LIBRARY_ALLOWLIST: dict[str, frozenset[str]] = {"mesh": frozenset({"trimesh."})}

# --- check 3 -----------------------------------------------------------------------------------

_WARP_SUFFIX = re.compile(r"\(Warp\)|on NVIDIA Warp")

# --- check 4 -----------------------------------------------------------------------------------

# Suite files that deliberately mirror no module.
_EXTRA_TEST_FILES = frozenset(
    {
        "parity",  # the cross-suite parity gate
        "api_conventions",  # this gate
        "map_uniform_probe",  # a Warp-behaviour probe, not a module's coverage
        "array_indexing_probe",  # ditto: Python-scope gather semantics, section 3.4
        "aggregate",  # covers benchmarks.aggregate, the loss-table loader -- tooling, not a module
        "launch",  # covers triwarp._launch, the private launcher every wrapper module calls
    }
)
_EXTRA_BENCHMARK_FILES = frozenset({"meshes"})  # mesh-fixture invariants, nothing timed

# Modules with nothing to test or nothing to time, each for a stated reason.
_MODULES_WITHOUT_TESTS = frozenset({"constants"})  # module-level ``wp.constant`` values only
_MODULES_WITHOUT_BENCHMARKS = frozenset(
    {
        "constants",  # no callables
        "typing",  # annotations and rank/dtype guards; nothing with a runtime cost
        "io",  # meshio round-trips, i.e. a benchmark of meshio
    }
)

# --- check 5 -----------------------------------------------------------------------------------

# Cross-module private imports this package accepts, as (importing module, ``owner._name``) pairs
# with the reason each one is not the defect the check is looking for. Anything else fires.
#
# The defect the check exists for is a private *operation* with callers in two modules -- a function
# that was written where it was first needed and should have been public
# (``proximity._default_mesh_query_max_dist``, four importers, one of them aliasing it back to a
# public-looking name). What survives below is the three things that are not that: a sorted copy
# whose public form does extra work, a string-to-kernel-flag table shared so two modules cannot
# disagree about one enum's spellings, and one pass a caller must skip.
#
# A cluster of entries sharing one (importer, owner) pair is a *misplaced family* rather than six
# exemptions: six such entries once sat here, all ``combine`` -> ``holes``, because the
# minimum-weight interval DP that fills one hole is the same DP that stitches two rims and the two
# halves lived in different modules. Moving the stitch family into ``holes`` deleted all six at
# once. Read a cluster that way before writing the next entry.
_PRIVATE_IMPORT_ALLOWLIST: dict[tuple[str, str], str] = {
    ("reduce", "array._sorted_copy"): (
        "reduce.median needs a sorted copy, and array.sort_and_argsort is the public form -- which "
        "also builds the permutation median throws away"
    ),
    ("remesh", "triangles._QUALITY_METRICS"): (
        "a string-to-kernel-flag table, for face_quality's metric argument, so remesh's objective "
        "names cannot drift from triangles.face_quality's"
    ),
    ("reconstruction", "remesh._flip_interior_edges"): (
        "ball pivoting flips its own output's interior edges; every public remesh entry point "
        "rebuilds connectivity as well, which reconstruction must not do to a fresh triangulation"
    ),
}

# --- check 6 -----------------------------------------------------------------------------------

# Duplicate public names that are correct in both vocabularies. ``array.concatenate`` /
# ``array.split`` mirror ``numpy.concatenate`` / ``numpy.split``, while ``combine.concatenate``
# mirrors ``trimesh.util.concatenate`` and ``combine.split`` mirrors ``trimesh.Trimesh.split``;
# none is reachable unqualified and the naming rule requires both spellings of each.
_DUPLICATE_NAME_ALLOWLIST: dict[str, frozenset[str]] = {
    "concatenate": frozenset({"array", "combine"}),
    "split": frozenset({"array", "combine"}),
}

# --- check 7 -----------------------------------------------------------------------------------

# Kernel-side libraries that back no single public module.
_SHARED_KERNEL_MODULES = frozenset({"predicates", "scatter"})

# Public modules that launch no kernel of their own and so have no kernel module.
_MODULES_WITHOUT_KERNELS = frozenset({"constants", "io", "mesh", "typing"})

# --- check 8 ------------------------------------------------------------------------------------

# Private helpers that sit above their first caller today. CLAUDE.md section 5 says a helper must
# never appear above the caller it serves, and the package had drifted off that rule wholesale
# before the check existed -- 49 sites across 15 modules, carried here as an explicit debt list
# rather than a silent exemption. That list has now been drained: one entry remains, and it is not
# debt but a permanent structural exemption. **Entries come out, they do not go in** -- a new helper
# must be placed correctly, which is what this check enforces.
#
# Two things the sweep learned, both of which the check itself caught:
#
# - Reordering has to consider *every* private helper in a module, not only the flagged ones.
#   Moving a flagged helper below a callee it uses strands that callee: ``holes._loop_perimeters``
#   was compliant before, because its caller sat above it, and became a fresh violation.
# - A name can appear several times at module scope, since ``@overload`` stubs precede their
#   implementation. Verifying a "pure move" by comparing functions keyed on name silently collapses
#   those duplicates, so the check on the *reorder* has to compare the multiset.
# --- check 9 ------------------------------------------------------------------------------------

# A Warp version claim, and the spelling is the convention: the word **Warp** immediately before
# the number. Anything looser is unusable here -- this package writes measured ratios in the same
# shape ("within 1.25x of best", "1.13x on CUDA at both sizes"), so a bare ``1.N`` token matches an
# order of magnitude more ratios than real claims. A version claim says "Warp 1.17", never
# "through 1.17" with the word three lines up.
_WARP_VERSION_CLAIM = re.compile(r"\bWarp\s+1\.(\d+)(?:\.(\d+))?\b")

# Version claims that deliberately record history rather than describe the installed Warp. Keyed by
# ``(module, "1.x")``; the value is the reason, and it is where the re-verification goes -- so the
# next upgrade reads a list of claims to re-run instead of a grep to invent. Modules under
# ``triwarp/`` are keyed by their dotted name, everything else by its path
# (``tests.api_conventions``, ``benchmarks.test_creation``).
_WARP_VERSION_ALLOWLIST: dict[tuple[str, str], str] = {
    # ---------------------------------------------------------------------------------------
    # Added by the Warp 1.17 upgrade. Every entry below is one of two things, and neither is a
    # claim about the installed Warp:
    #
    # * a **measurement stamp** on a recorded table -- the ratios were taken on 1.16 and were
    #   not re-run, so re-stamping them to 1.17 would assert a measurement nobody made. Re-run
    #   the table and *then* re-stamp, or leave the entry;
    # * genuine **history** -- the version in which a behaviour changed, which does not move
    #   when the installed version does.
    #
    # The mechanism claims that sat beside these were re-probed on 1.17 and re-stamped: the
    # CPU single-lane ``wp.launch_tiled`` (8.0 against 512.0 over 8 blocks of 64 ones,
    # unchanged), the empty-``wp.Mesh`` CUDA corruption (10/10 throwaway subprocesses still
    # abort), the ``radix_sort_pairs`` key-dtype set, a matrix's missing ``.shape`` in kernel
    # scope, ``wp.Volume``'s zero-point raise and the inclusive BVH box test.
    # ---------------------------------------------------------------------------------------
    ("kernels.holes", "1.16"): (
        "measurement stamp: the runtime-vs-constant stride A/B on a 400-vertex loop was taken "
        "on 1.16 and not re-run"
    ),
    ("kernels.points", "1.16"): (
        "measurement stamp: the ``count = 1024`` capture-and-replay comparison was taken on "
        "1.16 and not re-run"
    ),
    ("tests.test_voxels", "1.16"): (
        "deliberate history: names the version in which ``wp.Volume.allocate_by_voxels`` "
        "gained its CPU path, which is why this module is not CUDA-only"
    ),
    ("_device", "1.14"): (
        "deliberate history: the version in which ``wp.launch`` stopped validating a "
        "cross-device argument list (NVIDIA/warp GH-1461), which is why ``require_same_device`` "
        "exists at all and does not move when the installed version does"
    ),
}

# Directories check 9 scans, and the prefix each one's allowlist key carries. ``triwarp/`` keeps the
# bare dotted module name it has always used; the two suites are prefixed so a key stays unambiguous
# when a test file and a package module share a stem.
_WARP_VERSION_SCAN_ROOTS: tuple[tuple[Path, str], ...] = (
    (_PACKAGE_DIR, ""),
    (_TESTS_DIR, "tests."),
    (_BENCHMARKS_DIR, "benchmarks."),
)

# --- check 10 -----------------------------------------------------------------------------------

# The Python-scope allocators that take a ``device`` keyword. ``wp.clone`` and the ``*_like``
# family inherit the source array's device and so cannot get this wrong.
_ALLOCATORS = frozenset({"array", "empty", "full", "ones", "zeros"})

# The call prefixes check 10 and check 15 read: Warp's own allocators and launchers, and
# ``triwarp._launch``'s cached versions of them, which the wrapper layer calls instead and which
# take the same ``device`` keyword. Without the second, rewriting a call site onto the cached path
# would take it out of both checks' view with nothing going red.
_DEVICE_CALL_OWNERS = frozenset({("wp",), ("_launch",)})

# Allocations that deliberately fall back to Warp's current device, keyed by ``(module, call)``
# with the call spelled exactly as ``ast.unparse`` renders it. The value is the reason, and the
# docstring of the function it sits in has to say the same thing -- an undocumented fallback is
# the defect, not the fallback itself.
_ALLOCATION_DEVICE_ALLOWLIST: dict[tuple[str, str], str] = {
    ("combine", "_launch.empty(0, dtype=wp.vec3)"): (
        "concatenate([]) has no input to take a device from, so the current device is the only "
        "answer available; its Returns block says so"
    ),
    ("combine", "_launch.empty(0, dtype=wp.int32)"): (
        "the face half of the same empty return, for the same reason"
    ),
}

_HELPER_ORDER_ALLOWLIST: dict[str, frozenset[str]] = {
    # Not debt: ``_icosphere_face_table`` is called at *module* scope (line 179) to build the
    # ``_ICOSPHERE_FACE_TABLE`` constant, so it has no function caller to sit below and moving it
    # under its use is a NameError at import. This is the one permanent entry; the other 48 sites
    # this list carried were reordered in the section 4.4 sweep.
    "creation": frozenset({"_icosphere_face_table"})
}

# The Python-scope launchers that take a ``device`` keyword. Data for check 15, hoisted up here
# alongside check 10's device-related allowlists rather than beside check 15's own logic section.
_LAUNCHERS = frozenset({"launch", "launch_tiled"})

# Launches that deliberately fall back to Warp's current device, keyed by ``(module, kernel)``.
# Empty on purpose: every one of the package's launches forwards a device, and a new entry needs a
# reason a reader can check, because the failure mode is silent memory corruption on CPU runs.
_LAUNCH_DEVICE_ALLOWLIST: dict[tuple[str, str], str] = {}

# Kernel-scope calls whose first argument is a buffer the kernel writes. Data for check 13, hoisted
# up here for the same reason.
_KERNEL_WRITE_CALLS = frozenset(
    {
        "atomic_add",
        "atomic_sub",
        "atomic_min",
        "atomic_max",
        "atomic_cas",
        "atomic_exch",
        "tile_store",
        "tile_atomic_add",
    }
)

# Kernel arguments that are written without the ``out_`` prefix, or that legitimately follow an
# ``out_`` argument, keyed by ``(kernel module, kernel)``. Two exemption classes, both written into
# ``.claude/CLAUDE.md`` section 2.1:
#
# - **In-place**: the argument is both the input and the result -- an ``out_`` prefix would misread
#   as write-only. ``sort_rows_insertion(data)``,
#   ``offset_packed_faces(faces)``, the hole-filling DP tables (read at smaller spans, written at
#   the current one) and ``transform_and_accumulate_cost(acc)``, which reads the packed
#   accumulator's weight-sum slot while atomically adding into its cost slot.
# - **Scratch / persistent state**: caller-allocated working memory carried across launches --
#   cursors, stacks, open-addressing tables, the ear-clipping ring, ``ball_pivoting``'s
#   persistent front. Neither an input nor the answer, so the name says what the buffer holds
#   (``cursor``, ``front_out``, ``new_src``) rather than wearing a prefix that promises a result.
_KERNEL_OUTPUT_ALLOWLIST: dict[tuple[str, str], frozenset[str]] = {
    # in-place
    ("array", "sort_rows_insertion"): frozenset({"data"}),
    # ``neighbors`` is sorted in place: this kernel only orders the two slots each vertex already
    # holds, so it is both the input and the result and ``out_`` would read as write-only.
    # The refinement carries its chains across rounds: ``chains`` holds each chain's current frame
    # and ``chain_state`` its running loss and best box, so both are read, compared against and
    # conditionally overwritten by every round. They are the loop's state, not its answer -- the
    # answer is one row read back after the last round -- and ``out_`` would read as write-only.
    #
    # ``chain_state`` is deliberately absent: this scan resolves store targets syntactically, and
    # that buffer is now written only by ``write_chain_state``, a ``@wp.func`` the kernel hands it
    # to. So the check cannot see it as an output at all and an entry for it would be reported
    # stale. The convention still binds it -- it is in-place loop state, same as ``chains``.
    ("bounds", "oriented_box_select_chains"): frozenset({"chains"}),
    ("combine", "offset_packed_faces"): frozenset({"faces"}),
    # ``holes``' two DP tables are in place, and they ride inside ``HoleFillTables`` rather than in
    # the signature because their pointers are invariant across the whole span sweep and a launch
    # argument is not free (see that struct's docstring). So the bundle itself is the written
    # argument here, and ``out_tables`` would misread as write-only -- it is overwhelmingly
    # read-only inputs.
    # ``scatter_neighbor_lists``' ``cursor`` is the per-node write head the CSR fill hands out with
    # ``wp.atomic_add``: caller-allocated scratch, zeroed before the launch and meaningless after
    # it, so ``out_`` would advertise it as the answer -- the same role, and the same exemption, as
    # ``adjacency.scatter_vertex_faces``' argument of that name. ``bfs_push_level``'s ``state`` is
    # the level loop's persistent word -- the level to claim and the claim flag -- carried across
    # launches and both read and written every level, which is the ``forest_link`` case one kernel
    # down.
    ("graph", "scatter_neighbor_lists"): frozenset({"cursor"}),
    ("homology", "bfs_push_level"): frozenset({"state"}),
    # A Chronopoulos-Gear round updates its recurrence in place: ``p``, ``s = A p`` and ``r`` are
    # each read and overwritten by the one lane that owns the entry. They are the iteration's state,
    # not its answer (``x``, and the published scalars), so ``out_`` would read as write-only.
    # ``state`` -- the round loop's word, whose round count ``cg_update`` / ``cg_coefficients`` read
    # and whose condition slot they write -- is the same kind of argument and deliberately absent:
    # the write happens in ``cg_close_round``, a ``@wp.func`` this scan cannot see into.
    ("algorithms.conjugate_gradient", "cg_update"): frozenset({"p", "s", "r"}),
    # ``previous`` is a settle monitor's snapshot of the field at the last check, compared against
    # and then advanced to the current one: loop state carried across launches, not the answer.
    ("algorithms.conjugate_gradient", "cg_settle_change"): frozenset({"previous"}),
    # A settle monitor's verdict reads and advances its own state word (the last reached count, the
    # checks so far) and may clear the round loop's condition in ``state``, which the solve's own
    # launches also read and write: both are the loop's persistent state, not the answer.
    ("algorithms.conjugate_gradient", "cg_settle_decide"): frozenset({"settle", "state"}),
    # A one-launch solve's working vectors -- the residual, the search direction and their
    # images, the narrowed polynomial iterates -- are caller-allocated scratch, not the answer.
    ("algorithms.conjugate_gradient", "cg_one_block"): frozenset({"scratch", "narrow"}),
    # The heat method's normalization shifts and orients the distance field where it lies.
    ("heat", "shift_and_orient"): frozenset({"phi"}),
    # Boruvka's accepted edges leave the candidate set in place: what the caller reads afterwards
    # is the candidates the dual forest did not take, so ``candidate`` is input and result at once.
    ("homology", "forest_link"): frozenset({"candidate"}),
    ("holes", "fill_dp_span"): frozenset({"tables"}),
    ("holes", "fill_dp_span_tiled"): frozenset({"tables"}),
    # ``quadric_decimate``'s provenance column, folded one pass at a time: the array is the previous
    # pass's answer *and* this pass's, so it is in place and ``out_`` would read as write-only.
    ("remesh", "compact_decimation_pass"): frozenset({"index"}),
    # ``intrinsic_delaunay``'s halfedge-twin flip engine mutates the mesh it was handed rather than
    # producing a fresh one each round: ``edge_lengths`` is the caller's own metric, read pre-flip
    # and overwritten in the same launch, and ``twin`` is the incrementally-maintained twin table
    # every round both reads and updates. ``out_`` would misread either as write-only.
    #
    # ``faces`` is the same kind of in-place argument and is deliberately *not* listed: the kernel
    # now hands it to ``remesh.write_flipped_quad`` instead of storing into it directly, so this
    # check -- which resolves store targets syntactically -- no longer sees it written at all and
    # an entry here would match nothing. That is CLAUDE.md section 4.5's tension between the
    # shared-run extraction and a syntactic scan, resolved the way that section resolves it: the
    # extraction wins and the convention goes on binding the parameter unenforced. The staleness
    # half is what reported it, which is the argument for keeping that half.
    ("remesh", "commit_intrinsic_flips"): frozenset({"edge_lengths", "twin"}),
    # The other half of the same round's twin-table update: every halfedge not touched directly by
    # ``commit_intrinsic_flips`` reads and, where its neighbor moved, corrects its own twin pointer.
    ("remesh", "fixup_twin_remap"): frozenset({"twin"}),
    ("registration", "transform_and_accumulate_cost"): frozenset({"acc"}),
    # ``rhs`` arrives already holding ``-A_ub x_b`` from ``linalg.assemble_interior_system`` (which
    # eliminates a quadratic form with no linear term of its own), and this kernel only ever
    # accumulates the linear term on top -- the same in-place shape as ``acc`` just above, not a
    # fresh per-call answer.
    ("smoothing", "interior_mass_rhs_and_seed"): frozenset({"rhs"}),
    ("smoothing", "mut_dif_volume_correct"): frozenset({"slope"}),
    # ``filter_normals`` folds one pass's normalization into the next pass's seed, and both act on
    # the same accumulator slot: the thread reads its own entry and overwrites it in the same
    # launch, so ``out_`` would misread it as write-only. The answer is ``out_normals``.
    ("smoothing", "renormalize_and_reseed"): frozenset({"accumulated"}),
    # scratch / persistent state
    ("adjacency", "scatter_vertex_faces"): frozenset({"cursor"}),
    # ``min_distance_sq`` is the farthest-point sampler's running distance-to-the-chosen-set:
    # caller-allocated scratch the persistent block initializes and folds down one selection per
    # round. The answer is ``out_selected``; this buffer is the sweep's state.
    ("points", "farthest_point_sample_block"): frozenset({"min_distance_sq"}),
    # ``slot_count`` is the per-vertex atomic cursor picking which of the two neighbour slots each
    # scattered edge lands in -- the ``scatter_vertex_faces`` case exactly, under a name that says
    # what it holds. The answer is ``out_neighbors``.
    ("boundary", "scatter_boundary_neighbors"): frozenset({"slot_count"}),
    # ``table`` is ``voxels._VoxelTable``'s open-addressing cell table: filled with point indices
    # by the insert, rewritten to voxel rows by the ordering pass, then probed by the pooling
    # kernels. Neither an input nor the answer, so it keeps the name of what it holds.
    ("voxels", "insert_point_cells"): frozenset({"table"}),
    ("voxels", "assign_voxel_rows"): frozenset({"table"}),
    # The hull sweep's working set: the boundary polygon it carries between insertions, the buffer
    # it rebuilds that polygon into, and the per-boundary-edge orientations of one insertion.
    # Caller-allocated because the sweep is one thread over an ``n``-sized problem, so none of the
    # three can be a kernel local -- but none is an input or the answer either, so ``out_`` would
    # misread.
    ("reconstruction", "lexicographic_triangulation"): frozenset(
        {"boundary", "boundary_next", "orientations"}
    ),
    # ``edges`` is absent deliberately: this kernel mutates the table only through
    # ``register_face_edge``, and check 13 does not follow writes into a called ``@wp.func``.
    ("algorithms.ball_pivoting", "commit_triangles"): frozenset({"counters", "point_used"}),
    ("algorithms.ball_pivoting", "end_wave"): frozenset({"counters"}),
    # ``seed_failed`` is persistent per-point state, carried across every seeding wave for the
    # run's whole lifetime -- the ``point_used`` case one level up, not a fresh per-call answer.
    # ``seed_blocker`` is the same kind: the orphan each point last deferred to, a per-run cache.
    ("algorithms.ball_pivoting", "seed_triangles"): frozenset(
        {"counters", "seed_failed", "seed_blocker"}
    ),
    # ``edges`` is the open-addressing edge table (``BpaEdgeTable``): persistent state carried
    # across every wave, mutated in place by the pivot and commit kernels and read by both. It is
    # the ``front_out`` case one level up -- neither an input nor the answer -- so it keeps the name
    # of what it holds. Check 13 sees writes through its fields since ``_subscript_base`` resolves
    # ``edges.count[slot] = 1`` to ``edges``.
    ("algorithms.ball_pivoting", "pivot_front_edges"): frozenset(
        {"counters", "edges", "front_out"}
    ),
    ("algorithms.ball_pivoting", "rehash_edges"): frozenset(
        {"new_cand", "new_count", "new_opp", "new_src", "new_state", "new_tgt"}
    ),
    ("energies", "internal_angles_and_sums"): frozenset({"edge_cursor"}),
    ("repair", "emit_degree3_replacement"): frozenset({"cursor"}),
    # The flip loop's incremental state, carried across rounds and kept current by the commit that
    # rewrites the faces: the row <-> halfedge maps and the tracked valences. Neither an input nor
    # the answer (the answer is the face buffer), so they keep the names of what they hold.
    ("remesh", "commit_flips"): frozenset({"halfedge_row", "row_halfedges", "valence"}),
    ("repair", "emit_straighten_faces"): frozenset({"cursor"}),
    # ``counter`` and ``overflow`` are the work list the capped first pass hands the tiled second
    # one -- scratch, not the answer.
    #
    # ``global_best_sq`` -- the running minimum the two passes publish into and prune against -- is
    # the same exemption class and is deliberately **not** listed: both kernels now reach it only
    # through ``update_nearest_face_pair``, and this check resolves store targets syntactically, so
    # a buffer written inside a shared ``@wp.func`` is outside its view entirely. An entry for it
    # would be reported as matching nothing. That is CLAUDE.md section 4.5's documented tension
    # between this check and section 2.4's "extract the shared run", resolved the way it prescribes:
    # the extraction wins and the parameter goes on being scratch, unenforced. Do not re-add the
    # entry -- if the write ever comes back into a kernel body, add it again then.
    ("proximity", "face_to_mesh_distance"): frozenset({"counter", "overflow"}),
    ("grouping", "hash_insert"): frozenset({"slot_counts"}),
    # ``values`` is the matrix whose rows this scales -- input and result in the same buffer, since
    # the prolongation smoother's ``-w D^-1 (A P0)`` is a row scaling of a product that has just
    # been built and is not needed unscaled.
    ("algorithms.multigrid", "scale_rows"): frozenset({"values"}),
    # ``state`` is the ``wp.capture_while`` loop's own [rounds run, condition] pair, read and
    # incremented across launches -- the ``rdp_begin_round`` / ``rdp_split_spans`` case below,
    # under the same name.
    ("polyline", "ear_loop_continue"): frozenset({"state"}),
    # Round, pass 1 of 4 of the level-synchronous Ramer-Douglas-Peucker split: only arms the
    # per-span accumulators (``out_span_max`` / ``out_span_argmax``) and advances the loop's own
    # [rounds run, condition] pair -- the same ``state`` buffer ``rdp_split_spans`` and
    # ``ear_loop_continue`` carry under this name.
    ("polyline", "rdp_begin_round"): frozenset({"state"}),
    # The level-synchronous Ramer-Douglas-Peucker round. ``span_lo`` / ``span_hi`` are each point's
    # current span, rewritten in place to the child span it belongs to at the next level;
    # ``state`` is the ``wp.capture_while`` loop's own [levels run, condition] pair, which is the
    # ``ear_loop_continue`` case in the same module. Neither is an input and neither is the answer
    # -- that is ``out_keep``.
    ("polyline", "rdp_split_spans"): frozenset({"state"}),
    # The Poisson-disk elimination round's carried state: the deletions clear ``alive`` and take
    # their contributions off ``weights`` in the same pass, both rewritten in place every round.
    ("sample", "apply_deletions"): frozenset({"alive", "weights"}),
    # The min-weight fill's traceback walks each rim's predecessor table with an explicit stack of
    # pending intervals -- caller-allocated because the walk is one thread over a ``B``-deep
    # problem, so it cannot be a kernel local. Neither an input nor the answer, which is
    # ``out_triangles`` beside ``out_counts``.
    ("holes", "traceback_fill_triangles"): frozenset({"stack"}),
    ("visibility", "shape_diameter"): frozenset({"scratch"}),
}


@dataclass(frozen=True)
class PublicFunction:
    """A module-level ``def`` whose name does not start with an underscore."""

    module: str
    name: str
    lineno: int
    summary: str
    returns: str | None
    parameters: tuple[str, ...]
    raises: tuple[tuple[int, str], ...]
    """``(lineno, exception)`` for each ``raise`` in this function's own body, nested functions
    excluded."""

    documents_raises: bool
    """Whether the docstring carries a numpydoc ``Raises`` section header."""

    @property
    def site(self) -> str:
        """Render as ``path:lineno`` so editors and terminals can jump to it."""
        return f"triwarp/{self.module.replace('.', '/')}.py:{self.lineno}"


@dataclass(frozen=True)
class PublicModule:
    """One documented wrapper module: its dotted name, its summary and its public functions."""

    name: str
    path: str
    summary: str
    functions: tuple[PublicFunction, ...]
    private_imports: tuple[tuple[str, int], ...]
    """``("owner._name", lineno)`` for every private name reached across a module boundary."""

    early_helpers: tuple[tuple[str, int, int], ...]
    """``(name, def_lineno, first_caller_lineno)`` for each private helper defined above its
    first caller."""


@dataclass
class PackageScan:
    """Everything the checks below read, gathered in one pass over ``triwarp/``."""

    modules: dict[str, PublicModule] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def functions(self) -> list[PublicFunction]:
        """Every public function in the package, in module then source order."""
        return [function for module in self.modules.values() for function in module.functions]


def _summary(node: ast.Module | ast.FunctionDef) -> str:
    """Collapse a docstring's first paragraph onto one line, or ``""`` when there is none."""
    docstring = ast.get_docstring(node) or ""
    return " ".join(docstring.strip().split("\n\n")[0].split())


def _dotted(node: ast.expr) -> tuple[str, ...]:
    """Resolve a ``Name``/``Attribute`` chain to its dotted parts, or ``()`` for anything else."""
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return ()
    parts.append(current.id)
    return tuple(reversed(parts))


def _direct_raises(node: ast.FunctionDef) -> list[tuple[int, str]]:
    """
    Every ``raise`` in this function's own body, with the exception it names.

    A nested ``def``, ``lambda`` or ``class`` is not descended into -- its raises belong to *it*,
    and a closure's failure mode is its caller's docstring only by coincidence. A bare ``raise``
    re-raises something already in flight and names nothing, so it is skipped.
    """
    found: list[tuple[int, str]] = []
    stack: list[ast.AST] = list(ast.iter_child_nodes(node))
    while stack:
        current = stack.pop()
        if isinstance(current, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            continue
        if isinstance(current, ast.Raise) and current.exc is not None:
            exception = current.exc.func if isinstance(current.exc, ast.Call) else current.exc
            found.append((current.lineno, ast.unparse(exception)))
        stack.extend(ast.iter_child_nodes(current))
    return sorted(found)


_RAISES_SECTION = re.compile(r"^[ \t]*Raises[ \t]*\n[ \t]*-{5,}[ \t]*$", re.MULTILINE)


def _documents_raises(node: ast.FunctionDef) -> bool:
    """Whether the docstring has a numpydoc ``Raises`` header, underline and all."""
    return _RAISES_SECTION.search(ast.get_docstring(node, clean=False) or "") is not None


def _bare_allocations(tree: ast.Module) -> list[tuple[str, int]]:
    """
    Python-scope ``wp.<allocator>(...)`` calls that name no ``device``.

    A ``**kwargs`` splat could be carrying one, so a call with one is not reported -- the check
    would rather miss a forwarded device than fire on a call it cannot read.
    """
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in _ALLOCATORS or _dotted(node.func)[:1] not in _DEVICE_CALL_OWNERS:
            continue
        if any(keyword.arg in ("device", None) for keyword in node.keywords):
            continue
        found.append((ast.unparse(node), node.lineno))
    return sorted(found, key=lambda item: item[1])


def _private_imports(tree: ast.Module, module: str) -> list[tuple[str, int]]:
    """
    Every private name this module reaches for across a module boundary.

    Two spellings reach one: ``from triwarp.x import _y`` (which ``ray.py`` used to alias straight
    back to a public-looking name) and the attribute form ``tw.x._y``. Both are collected as
    ``"x._y"`` so the allowlist is keyed the same way regardless of which was used.
    """
    found: dict[tuple[str, int], None] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("triwarp"):
            owner = node.module.removeprefix("triwarp").lstrip(".")
            if owner in ("", module) or owner.startswith("kernels"):
                continue
            for alias in node.names:
                if alias.name.startswith("_"):
                    found[(f"{owner}.{alias.name}", node.lineno)] = None
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            parts = _dotted(node)
            if len(parts) == 3 and parts[0] == "tw" and parts[1] != module:
                found[(f"{parts[1]}.{parts[2]}", node.lineno)] = None
    return sorted(found)


def _early_helpers(tree: ast.Module) -> list[tuple[str, int, int]]:
    """
    Private module-level helpers whose ``def`` precedes their first reference in the same module.

    A reference inside the helper's own body is skipped, so recursion and a self-referencing
    closure do not count as callers. A helper with no caller at all is not reported here -- check 5
    already covers the cross-module case, and a genuinely dead helper is a different defect.
    """
    definitions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("_")
        and not node.name.startswith("__")
    }
    first_use: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
            continue
        definition = definitions.get(node.id)
        if definition is None:
            continue
        if definition.lineno <= node.lineno <= (definition.end_lineno or definition.lineno):
            continue
        first_use[node.id] = min(first_use.get(node.id, node.lineno), node.lineno)
    return sorted(
        (name, definition.lineno, first_use[name])
        for name, definition in definitions.items()
        if name in first_use and definition.lineno < first_use[name]
    )


@functools.cache
def scan_package() -> PackageScan:
    """Read every public wrapper module under ``triwarp/``, skipping ``kernels/`` and ``_*.py``."""
    scan = PackageScan()
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        relative = path.relative_to(_PACKAGE_DIR)
        if relative.parts[0] == "kernels" or path.name == "__init__.py":
            continue
        if any(part.startswith("_") for part in relative.with_suffix("").parts):
            continue
        module = ".".join(relative.with_suffix("").parts)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as error:
            scan.errors.append(f"triwarp/{relative.as_posix()}:{error.lineno or 0}: {error.msg}")
            continue

        functions = tuple(
            PublicFunction(
                module=module,
                name=node.name,
                lineno=node.lineno,
                summary=_summary(node),
                returns=ast.unparse(node.returns) if node.returns is not None else None,
                parameters=tuple(
                    argument.arg
                    for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
                ),
                raises=tuple(_direct_raises(node)),
                documents_raises=_documents_raises(node),
            )
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
        )
        scan.modules[module] = PublicModule(
            name=module,
            path=f"triwarp/{relative.as_posix()}",
            summary=_summary(tree),
            functions=functions,
            private_imports=tuple(_private_imports(tree, module)),
            early_helpers=tuple(_early_helpers(tree)),
        )
    return scan


def library_in_summary_problems() -> list[str]:
    """Check 1: a public summary line names a reference library."""
    problems: list[str] = []
    for module in scan_package().modules.values():
        allowed = _SUMMARY_LIBRARY_ALLOWLIST.get(module.name, frozenset())
        for where, summary in (
            (f"{module.path}:1 <module>", module.summary),
            *((f"{item.site} {item.name}", item.summary) for item in module.functions),
        ):
            match = _LIBRARY_IN_SUMMARY.search(summary)
            if match is not None and match.group() not in allowed:
                problems.append(
                    f"{where}: summary names {match.group()!r} -- say what it returns and move the "
                    f"attribution to Notes or See Also: {summary!r}"
                )
    return problems


def mask_return_problems() -> list[str]:
    """Check 2: a ``*_mask`` producer whose return annotation is not a boolean array."""
    problems: list[str] = []
    for function in scan_package().functions:
        if not function.name.endswith("_mask"):
            continue
        # A function taking a mask is named for its *input* (``submesh_from_face_mask``), so the
        # suffix says nothing about what it returns.
        if any(parameter.endswith("mask") for parameter in function.parameters):
            continue
        if function.returns is not None and "wp.bool" in function.returns:
            continue
        problems.append(
            f"{function.site} {function.name}: returns {function.returns} -- every other member of "
            "the *_mask family returns wp.array[wp.bool]; rename it out of the family"
        )
    return problems


def warp_suffix_problems() -> list[str]:
    """Check 3: a module summary advertising the framework the whole package is built on."""
    return [
        f"{module.path}:1 <module>: summary carries a Warp suffix; the whole package is Warp: "
        f"{module.summary!r}"
        for module in scan_package().modules.values()
        if _WARP_SUFFIX.search(module.summary)
    ]


def coverage_location_problems() -> list[str]:
    """Check 4: a suite file that mirrors no module, or a module with no file in a suite."""
    modules = set(scan_package().modules)
    problems: list[str] = []
    for directory, extras, exempt in (
        (_TESTS_DIR, _EXTRA_TEST_FILES, _MODULES_WITHOUT_TESTS),
        (_BENCHMARKS_DIR, _EXTRA_BENCHMARK_FILES, _MODULES_WITHOUT_BENCHMARKS),
    ):
        if not directory.is_dir():
            continue
        suite = directory.name
        stems = {path.stem.removeprefix("test_") for path in directory.glob("test_*.py")}
        for stem in sorted(stems - modules - extras):
            problems.append(
                f"{suite}/test_{stem}.py: mirrors no module -- name it for the one it covers"
            )
        for module in sorted(modules - stems - exempt):
            problems.append(
                f"triwarp/{module}.py: no {suite}/test_{module}.py -- coverage is per module"
            )
    return problems


def private_import_problems() -> list[str]:
    """Check 5: a private name reached across a module boundary."""
    problems: list[str] = []
    for module in scan_package().modules.values():
        for target, lineno in module.private_imports:
            if (module.name, target) in _PRIVATE_IMPORT_ALLOWLIST:
                continue
            problems.append(
                f"{module.path}:{lineno}: reaches {target} in another module -- a helper with "
                "callers in two modules is a public function, not a private one"
            )
    return problems


def duplicate_name_problems() -> list[str]:
    """Check 6: one public name exported by two modules."""
    homes: dict[str, set[str]] = defaultdict(set)
    for function in scan_package().functions:
        homes[function.name].add(function.module)
    return [
        f"{name!r} is public in {', '.join(sorted(modules))} -- two functions with one name are "
        "one name too few; say what each returns"
        for name, modules in sorted(homes.items())
        if len(modules) > 1 and frozenset(modules) != _DUPLICATE_NAME_ALLOWLIST.get(name)
    ]


def kernel_module_problems() -> list[str]:
    """Check 7: a top-level kernel module and its public module must be named for each other."""
    public = {path.stem for path in _PACKAGE_DIR.glob("*.py") if path.stem != "__init__"}
    public -= {stem for stem in public if stem.startswith("_")}
    kernels = {path.stem for path in _KERNELS_DIR.glob("*.py") if path.stem != "__init__"}
    problems = [
        f"triwarp/kernels/{stem}.py: no triwarp/{stem}.py -- a top-level kernel module is named "
        "for the public module it backs"
        for stem in sorted(kernels - public - _SHARED_KERNEL_MODULES)
    ]
    problems += [
        f"triwarp/{stem}.py: no triwarp/kernels/{stem}.py -- a moved wrapper takes its kernels "
        "with it"
        for stem in sorted(public - kernels - _MODULES_WITHOUT_KERNELS)
    ]
    return problems


def helper_order_problems() -> list[str]:
    """Check 8: a private helper defined above its first caller (the stepdown rule)."""
    problems: list[str] = []
    for module in scan_package().modules.values():
        allowed = _HELPER_ORDER_ALLOWLIST.get(module.name, frozenset())
        for name, def_line, use_line in module.early_helpers:
            if name in allowed:
                continue
            problems.append(
                f"{module.path}:{def_line}: {name!r} is defined above its first caller "
                f"(line {use_line}) -- a private helper goes immediately after the public function "
                "that calls it, or after its last caller when several do"
            )
    stale = [
        f"{module.path}: {name!r} is in _HELPER_ORDER_ALLOWLIST but is no longer out of order -- "
        "drop the entry, the debt list only shrinks"
        for module in scan_package().modules.values()
        for name in sorted(_HELPER_ORDER_ALLOWLIST.get(module.name, frozenset()))
        if name not in {helper[0] for helper in module.early_helpers}
    ]
    return problems + stale


def installed_warp_version() -> str | None:
    """Report the installed ``warp-lang`` version, or ``None`` when the distribution is absent."""
    try:
        return installed_version("warp-lang")
    except PackageNotFoundError:
        return None


def _prose_blocks(source: str) -> list[tuple[int, str]]:
    """
    ``(lineno, text)`` for each run of consecutive comment lines and each string literal.

    Newlines are preserved inside a block so a caller can recover the exact line from a match
    offset. String literals cover every docstring without walking the tree, and a version claim
    inside a non-docstring string is still a claim.
    """
    blocks: list[tuple[int, str]] = []
    run_start, run_lines = 0, []
    for lineno, line in enumerate(source.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            if not run_lines:
                run_start = lineno
            run_lines.append(stripped.lstrip("#"))
            continue
        if run_lines:
            blocks.append((run_start, "\n".join(run_lines)))
            run_lines = []
    if run_lines:
        blocks.append((run_start, "\n".join(run_lines)))

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return blocks
    blocks.extend(
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )
    return blocks


def warp_version_problems() -> list[str]:
    """Check 9: a Warp-version claim naming a version older than the installed ``warp-lang``."""
    installed = installed_warp_version()
    if installed is None:  # nothing to compare against; the check abstains rather than guesses
        return []
    installed_minor = int(installed.split(".")[1])

    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    for root, prefix in _WARP_VERSION_SCAN_ROOTS:
        if not root.is_dir():  # neither suite ships in the wheel
            continue
        for path in sorted(root.rglob("*.py")):
            module = path.relative_to(root).with_suffix("").as_posix().replace("/", ".")
            module = prefix + module.removesuffix(".__init__").lstrip(".")
            for lineno, text in _prose_blocks(path.read_text()):
                for match in _WARP_VERSION_CLAIM.finditer(text):
                    line = lineno + text.count("\n", 0, match.start())
                    minor = int(match.group(1))
                    if minor >= installed_minor:
                        continue
                    key = (module, f"1.{minor}")
                    if key in _WARP_VERSION_ALLOWLIST:
                        seen.add(key)
                        continue
                    problems.append(
                        f"{path.relative_to(_REPO_ROOT)}:{line}: names {match.group(0)}, "
                        f"older than the installed warp-lang {installed} -- re-probe the claim "
                        "against the installed version and re-stamp it, or add a "
                        "_WARP_VERSION_ALLOWLIST entry recording why it deliberately names history"
                    )
    problems.extend(
        f"_WARP_VERSION_ALLOWLIST entry {key!r} matches nothing in the scanned tree -- drop it"
        for key in sorted(_WARP_VERSION_ALLOWLIST)
        if key not in seen and int(key[1].split(".")[1]) < installed_minor
    )
    return problems


def allocation_device_problems() -> list[str]:
    """
    Check 10: a Python-scope allocation that does not name the device it allocates on.

    Walks the whole package rather than ``scan_package``'s public subset -- private ``_*.py``
    modules allocate too, and a buffer landing on the wrong device is not a question about the
    API's shape. ``kernels/`` is excluded because a kernel allocates nothing at Python scope.
    """
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        relative = path.relative_to(_PACKAGE_DIR)
        if relative.parts[0] == "kernels":
            continue
        module = ".".join(relative.with_suffix("").parts).removesuffix(".__init__")
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue  # test_package_scan_is_discoverable reports the parse failure
        for call, lineno in _bare_allocations(tree):
            key = (module, call)
            if key in _ALLOCATION_DEVICE_ALLOWLIST:
                seen.add(key)
                continue
            problems.append(
                f"triwarp/{relative.as_posix()}:{lineno}: {call} has no device= -- it lands on "
                "Warp's *current* device, which is the input's only by accident under the test "
                "suite; forward the device of the arrays it will be used with"
            )
    problems.extend(
        f"_ALLOCATION_DEVICE_ALLOWLIST entry {key!r} matches nothing in triwarp/ -- drop it"
        for key in sorted(_ALLOCATION_DEVICE_ALLOWLIST)
        if key not in seen
    )
    return problems


def undocumented_raise_problems() -> list[str]:
    """
    Check 11: a public function with a direct ``raise`` and no ``Raises`` block.

    One direction only. The reverse -- a ``Raises`` block with no direct raise -- is the *correct*
    shape for the 44 functions that delegate validation to a shared guard, and scanning it would
    need an allowlist longer than the check.
    """
    return [
        f"{function.site} {function.name}: raises "
        f"{', '.join(sorted({exception for _, exception in function.raises}))} at line(s) "
        f"{', '.join(str(lineno) for lineno, _ in function.raises)} but documents no Raises block"
        for function in scan_package().functions
        if function.raises and not function.documents_raises
    ]


# --- check 12 -----------------------------------------------------------------------------------

_PYTHON_FENCE = re.compile(r"^(?P<indent>[ \t]*)```python[ \t]*$", re.MULTILINE)


@dataclass(frozen=True)
class DocstringExample:
    """One fenced ``python`` block lifted out of a docstring, dedented and ready to ``exec``."""

    site: str
    code: str

    @property
    def is_sketch(self) -> bool:
        """
        Whether the block is a deliberate outline: a bare ``...`` standing in for real code.

        Read as an ``Ellipsis`` *statement* rather than by matching the text, so a trailing comment
        (``...  # rewrite rhs in place``) still counts and a genuine ``...`` inside an expression
        does not.
        """
        try:
            tree = ast.parse(self.code)
        except SyntaxError:
            return False
        return any(
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and node.value.value is Ellipsis
            for node in ast.walk(tree)
        )


def docstring_examples() -> list[DocstringExample]:
    """
    Every fenced ``python`` block in ``triwarp/``, dedented to column zero.

    Read off the raw source rather than off docstring nodes: a block's *line number* is what makes
    a failure reportable, and ``ast`` gives the docstring's line, not the fence's. ``kernels/`` is
    excluded -- it has no fenced blocks, and kernel-scope code could not be executed as written.
    """
    examples: list[DocstringExample] = []
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        relative = path.relative_to(_PACKAGE_DIR)
        if relative.parts[0] == "kernels":
            continue
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        for match in _PYTHON_FENCE.finditer(source):
            indent = match.group("indent")
            fence = source.count("\n", 0, match.start())  # 0-based index of the ```python line
            body: list[str] = []
            for line in lines[fence + 1 :]:
                if line.strip() == "```":
                    break
                body.append(line.removeprefix(indent))
            examples.append(
                DocstringExample(
                    site=f"triwarp/{relative.as_posix()}:{fence + 1}", code="\n".join(body) + "\n"
                )
            )
    return examples


# --- check 13 -----------------------------------------------------------------------------------


def _factory_registered(tree: ast.Module, attribute: str) -> frozenset[str]:
    """
    Names passed as the first argument to ``wp.<attribute>(...)`` anywhere in ``tree``.

    A kernel factory (``.claude/CLAUDE.md`` section 2.7's ``wp.kernel(_k, name=...)``) registers a
    plain nested ``def`` whose body is Warp's DSL but which carries **no decorator**, so every
    check that enumerates kernel scope by walking ``decorator_list`` is blind to it. That is
    section 4's "a new Warp construct can silently switch off a static check" with the construct
    being the factory: measured when this was added, **14** such bodies existed and **six** carried
    defects the family of checks 16-22 exists to catch -- four bare ``wp.tid()`` and two
    kernel-scope ternaries, in ``kernels/reduce.py`` and ``kernels/neighbors.py``, the two modules
    that use factories most.

    Resolved by *name* rather than by identity because the registration
    (``return wp.kernel(_k, name=name)``) is textually separate from the ``def``; a factory that
    ever shadowed one nested name with another kernel-scope body in the same module would over-
    report, which no module does and which would be a defect of its own.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        target = node.func
        if (
            isinstance(target, ast.Attribute)
            and target.attr == attribute
            and isinstance(target.value, ast.Name)
            and target.value.id == "wp"
            and isinstance(node.args[0], ast.Name)
        ):
            names.add(node.args[0].id)
    return frozenset(names)


def _is_kernel(node: ast.FunctionDef, tree: ast.Module | None = None) -> bool:
    """
    Whether ``node`` is a kernel: decorated ``@wp.kernel`` / ``@wp.kernel(...)``, or factory-built.

    ``tree`` enables the factory half; pass it unless the caller has already filtered to decorated
    definitions for a reason it can state.
    """
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if (
            isinstance(target, ast.Attribute)
            and target.attr == "kernel"
            and isinstance(target.value, ast.Name)
            and target.value.id == "wp"
        ):
            return True
    return tree is not None and node.name in _factory_registered(tree, "kernel")


def _subscript_base(node: ast.expr) -> str | None:
    """
    Resolve a (possibly nested) subscript target to its bare name, or ``None``.

    A ``@wp.struct`` field counts as its struct: ``edges.count[slot] = 1`` resolves to ``edges``.
    Without that, bundling a kernel's buffers into a struct would make every write through them
    invisible to check 13 -- which is not hypothetical, it is what happened the first time
    ``ball_pivoting``'s edge table was bundled.
    """
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _written_parameters(node: ast.FunctionDef) -> set[str]:
    """
    Parameters the kernel body writes *directly*.

    A subscript store, or the first argument of a ``wp.atomic_*`` / ``wp.tile_store`` /
    ``wp.tile_atomic_add`` call, in either case through a ``@wp.struct`` field as well as directly.
    A write that happens inside a called ``@wp.func`` (e.g. through a ``wp.ref`` parameter) is
    invisible here, so this check can under-report but never false-positive.
    """
    parameters = {arg.arg for arg in node.args.args}
    written: set[str] = set()
    for sub in ast.walk(node):
        targets: list[ast.expr] = []
        if isinstance(sub, ast.Assign):
            targets = list(sub.targets)
        elif isinstance(sub, ast.AugAssign):
            targets = [sub.target]
        for target in targets:
            if isinstance(target, ast.Subscript):
                base = _subscript_base(target)
                if base in parameters:
                    written.add(base)
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr in _KERNEL_WRITE_CALLS
            and isinstance(sub.func.value, ast.Name)
            and sub.func.value.id == "wp"
            and sub.args
            and _subscript_base(sub.args[0]) in parameters
        ):
            base = _subscript_base(sub.args[0])
            assert base is not None
            written.add(base)
    return written


def kernel_output_naming_problems() -> list[str]:
    """
    Check 13: a kernel output argument without the ``out_`` prefix, or not at the signature's end.

    ``.claude/CLAUDE.md`` section 2.1's naming rule for ``triwarp/kernels/``, with its two written
    exemptions (in-place arguments and scratch / persistent-state buffers) carried by
    ``_KERNEL_OUTPUT_ALLOWLIST``. Both directions are checked: a *written* argument must wear the
    prefix, and nothing without the prefix may follow the first argument that wears it.
    """
    problems: list[str] = []
    seen: set[tuple[tuple[str, str], str]] = set()
    for path in sorted(_KERNELS_DIR.rglob("*.py")):
        module = ".".join(path.relative_to(_KERNELS_DIR).with_suffix("").parts)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue  # test_package_scan_is_discoverable reports the parse failure
        # ``ast.walk``, not ``tree.body``: a kernel factory's body is a *nested* ``def``, so a
        # top-level pass never sees it. It reached the tree as ``out`` where the convention is
        # ``out_`` and nothing complained -- see ``_factory_registered``.
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or not _is_kernel(node, tree):
                continue
            allowed = _KERNEL_OUTPUT_ALLOWLIST.get((module, node.name), frozenset())
            site = f"{path.relative_to(_REPO_ROOT)}:{node.lineno}"
            for name in sorted(_written_parameters(node)):
                if name.startswith("out_"):
                    continue
                if name in allowed:
                    seen.add(((module, node.name), name))
                    continue
                problems.append(
                    f"{site} {node.name}: writes parameter '{name}', which is neither "
                    "out_-prefixed nor an allowlisted in-place / scratch argument"
                )
            past_outputs = False
            for arg in node.args.args:
                if arg.arg.startswith("out_"):
                    past_outputs = True
                elif past_outputs:
                    if arg.arg in allowed:
                        seen.add(((module, node.name), arg.arg))
                        continue
                    problems.append(
                        f"{site} {node.name}: input parameter '{arg.arg}' follows an out_ "
                        "argument -- outputs go last"
                    )
    problems.extend(
        f"_KERNEL_OUTPUT_ALLOWLIST entry {(*key, name)!r} matches nothing in kernels/ -- drop it"
        for key, names in sorted(_KERNEL_OUTPUT_ALLOWLIST.items())
        for name in sorted(names)
        if (key, name) not in seen
    )
    return problems


# --- check 14 -----------------------------------------------------------------------------------


def _annotations(tree: ast.Module) -> list[ast.expr]:
    """Every annotation expression in a module: parameters, returns, and annotated assignments."""
    found: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            found.extend(arg.annotation for arg in arguments if arg.annotation is not None)
            if node.returns is not None:
                found.append(node.returns)
        elif isinstance(node, ast.AnnAssign):
            found.append(node.annotation)
    return found


def array_annotation_style_problems() -> list[str]:
    """
    Check 14: an array annotation spelled ``wp.array(dtype=T)`` rather than ``wp.array[T]``.

    ``.claude/CLAUDE.md`` section 1.2's subscript style, restricted to *annotation* positions so the
    scan can cover the whole package: at Python scope ``wp.array(dtype=T)`` is also a legal
    allocation expression, and only in an annotation is it the pre-1.12 spelling.

    Both forms work, which is why the old one survived in the three newest large kernel modules
    (``algorithms/ball_pivoting.py``, ``reconstruction.py``, ``remesh.py``) long after the
    convention settled -- 176 annotations against 1 644 in the current style, and ``remesh.py``
    mixed the two inside one file, which is the state that leaves a reader unsure which is current.
    """
    problems: list[str] = []
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue  # test_package_scan_is_discoverable reports the parse failure
        for annotation in _annotations(tree):
            for node in ast.walk(annotation):
                if not isinstance(node, ast.Call):
                    continue
                target = ".".join(_dotted(node.func))
                if target.startswith(("wp.array", "warp.array")):
                    problems.append(
                        f"{path.relative_to(_REPO_ROOT)}:{node.lineno} annotation "
                        f"'{ast.unparse(node)}' uses the pre-1.12 call style -- write it as "
                        f"{target}[...] instead"
                    )
    return problems


# --- check 15 -----------------------------------------------------------------------------------


def launch_device_problems() -> list[str]:
    """
    Check 15: a ``wp.launch`` / ``wp.launch_tiled`` that does not name the device it launches on.

    An omitted ``device=`` resolves to Warp's *current* device, which is ``cuda:0`` whenever CUDA is
    present. When the arrays are on the CPU the launch is not rejected and not wrong: Warp permits
    it by design on a system whose GPU can address host memory (``RELAXED`` and ``CHECKED`` both
    pass it), and the kernel returns the correct answer. What it does not get is ordering, so the
    host buffers are freed at scope exit while the kernel is still reading them, and the process
    aborts later inside glibc -- measured on the reduced reproducer at 20/20 aborts with a free and
    no synchronize against 0/20 with either. ``vertices.mean_vertex_normals`` shipped this way.

    This is the static half of the guard and it is the half that carries the load. The runtime half
    (``tests/conftest.py`` sets ``LaunchArrayAccessMode.STRICT``) only bites when the arrays are not
    on the launch device, so on a CUDA run -- what CI and this box do -- the default device *is* the
    arrays' device and the omission stays invisible. Only the scan sees it there.
    """
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        relative = path.relative_to(_PACKAGE_DIR)
        module = ".".join(relative.with_suffix("").parts).removesuffix(".__init__")
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue  # test_package_scan_is_discoverable reports the parse failure
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in _LAUNCHERS:
                continue
            if _dotted(node.func)[:1] not in _DEVICE_CALL_OWNERS:
                continue
            if any(keyword.arg in ("device", None) for keyword in node.keywords):
                continue  # a **kwargs splat may be forwarding one; miss it rather than misfire
            kernel = ast.unparse(node.args[0]) if node.args else node.func.attr
            key = (module, kernel)
            if key in _LAUNCH_DEVICE_ALLOWLIST:
                seen.add(key)
                continue
            problems.append(
                f"triwarp/{relative.as_posix()}:{node.lineno}: wp.{node.func.attr}({kernel}, ...) "
                "has no device= -- it launches on Warp's *current* device, so with CPU arrays it "
                "runs the kernel on cuda:0 over host pointers, returns the right answer, and "
                "corrupts the host heap when those arrays are freed mid-kernel; forward the device "
                "of the input arrays"
            )
    problems.extend(
        f"_LAUNCH_DEVICE_ALLOWLIST entry {key!r} matches nothing in triwarp/ -- drop it"
        for key in sorted(_LAUNCH_DEVICE_ALLOWLIST)
        if key not in seen
    )
    return problems


# --- check 16 -----------------------------------------------------------------------------------

_BUILTIN_CASTS = frozenset({"int", "float"})


def _kernel_scope_functions(tree: ast.Module) -> list[ast.FunctionDef]:
    """
    Every ``@wp.kernel`` / ``@wp.func`` in a module, whose body is Warp's DSL and not Python.

    Includes the **factory-registered** bodies -- a nested ``def`` handed to ``wp.kernel(...)`` or
    ``wp.func(...)`` rather than decorated -- which are kernel scope with no decorator to match on;
    see [`_factory_registered`][tests.api_conventions._factory_registered] for what that blind spot
    was hiding. The enclosing *factory* is ordinary Python and stays out of scope, which is what
    keeps a ``row_size: int`` parameter from tripping check 18.
    """
    factory = _factory_registered(tree, "kernel") | _factory_registered(tree, "func")
    found: list[ast.FunctionDef] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        decorators = " ".join(ast.unparse(decorator) for decorator in node.decorator_list)
        if "wp.kernel" in decorators or "wp.func" in decorators or node.name in factory:
            found.append(node)
    return found


def builtin_cast_problems() -> list[str]:
    """
    Check 16: a bare ``int(...)`` / ``float(...)`` inside a ``@wp.kernel`` or ``@wp.func`` body.

    ``.claude/CLAUDE.md`` section 1.3: ``wp.int32`` / ``wp.float32`` is the tree's only cast
    spelling. ``int`` and ``float`` are the same Warp builtins under a different name -- ``int(x)``
    compiles only because Warp writes an unconditional ``#define int(x) cast_int(x)`` into every
    generated module header, ``wp::int(x)`` not being valid C++ -- and the generated code is
    identical.

    The asymmetry that makes this a rule rather than a preference is ``float``: inside a
    ``wp.Float``-generic function ``total / float(count)`` does not narrow silently, it *fails to
    parse* (``Input types must be the same, got ['float64', 'float32']``). So every bare ``float()``
    is an unannounced decision that its function will never be generic, against section 14's
    "prefer dtype-generic ``@wp.func``s". Where the function is or could be generic the spelling is
    ``type(x)(...)``, the ``kernels/predicates.py`` convention.

    Scoped to kernel-scope bodies because at Python scope ``int(...)`` / ``float(...)`` are the
    ordinary builtins and entirely correct -- ``wp.constant(wp.float32(float("nan")))`` at module
    scope in ``kernels/texture.py`` is a Python call and is not a violation.
    """
    problems: list[str] = []
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue  # test_package_scan_is_discoverable reports the parse failure
        for function in _kernel_scope_functions(tree):
            for node in ast.walk(function):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                if node.func.id not in _BUILTIN_CASTS:
                    continue
                replacement = "wp.int32" if node.func.id == "int" else "wp.float32"
                problems.append(
                    f"{path.relative_to(_REPO_ROOT)}:{node.lineno} {function.name} casts with "
                    f"'{ast.unparse(node)}' -- write {replacement}(...) instead, or type(x)(...) "
                    "if the enclosing function is or could be dtype-generic"
                )
    return problems


# --- check 17 -----------------------------------------------------------------------------------

_INT_SCALARS = frozenset(
    {
        "wp.int8",
        "wp.int16",
        "wp.int32",
        "wp.int64",
        "wp.uint8",
        "wp.uint16",
        "wp.uint32",
        "wp.uint64",
        "wp.Int",
    }
)

# Binary operators that keep an integer integral. ``Div`` is deliberately absent: its result is the
# thing under test, so admitting it would let one unflagged division launder its operands into a
# second.
_INT_PRESERVING = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.BitAnd, ast.BitOr)


def _int_array_dtype(annotation: str) -> bool:
    """Whether a ``wp.array[...]`` annotation names a concrete integer dtype."""
    inside = annotation.partition("[")[2].rpartition("]")[0]
    return inside.split(",")[0].strip() in _INT_SCALARS


def _int_module_constants(tree: ast.Module) -> set[str]:
    """Module-level ``wp.constant(wp.int32(...))`` names, which kernels read as plain integers."""
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        value = node.value
        if not isinstance(target, ast.Name) or not isinstance(value, ast.Call):
            continue
        if _dotted(value.func) != ("wp", "constant") or not value.args:
            continue
        inner = value.args[0]
        if isinstance(inner, ast.Call) and ".".join(_dotted(inner.func)) in _INT_SCALARS:
            names.add(target.id)
    return names


def _int_typed_names(function: ast.FunctionDef, constants: set[str]) -> tuple[set[str], set[str]]:
    """
    Names inside a kernel-scope body that are integers *by declaration*, and integer arrays.

    Declaration, not inference: a module-level integer ``wp.constant``, an annotated parameter, or
    a local whose right-hand side is a ``wp.tid()``, an integer constructor, a ``.shape[...]``, or
    an integer-preserving expression over names already known. Anything the scan cannot type this
    way stays untyped, so a division involving it is never reported -- the check misses rather than
    misfires.
    """
    scalars: set[str] = set(constants)
    arrays: set[str] = set()
    for argument in function.args.args:
        if argument.annotation is None:
            continue
        annotation = ast.unparse(argument.annotation)
        if annotation in _INT_SCALARS:
            scalars.add(argument.arg)
        elif annotation.startswith("wp.array") and _int_array_dtype(annotation):
            arrays.add(argument.arg)
    # Two passes so an assignment reached before its operands were typed still resolves; the
    # dependency chains here are short and a third pass has never added a name.
    for _ in range(2):
        for node in ast.walk(function):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Tuple):
                # ``i, j = wp.tid()``: every element of a multi-dimensional thread index is an int.
                value = node.value
                if isinstance(value, ast.Call) and _dotted(value.func) == ("wp", "tid"):
                    scalars.update(
                        element.id for element in target.elts if isinstance(element, ast.Name)
                    )
                continue
            if isinstance(target, ast.Name) and _is_int_expression(node.value, scalars, arrays):
                scalars.add(target.id)
    return scalars, arrays


def _is_int_expression(node: ast.expr, scalars: set[str], arrays: set[str]) -> bool:
    """Whether ``node`` is an integer by declaration, given the names already typed."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int) and not isinstance(node.value, bool)
    if isinstance(node, ast.Name):
        return node.id in scalars
    if isinstance(node, ast.Call):
        dotted = _dotted(node.func)
        return ".".join(dotted) in _INT_SCALARS or dotted == ("wp", "tid")
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Attribute) and node.value.attr == "shape":
            return True
        return isinstance(node.value, ast.Name) and node.value.id in arrays
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_int_expression(node.operand, scalars, arrays)
    if isinstance(node, ast.BinOp) and isinstance(node.op, _INT_PRESERVING):
        return _is_int_expression(node.left, scalars, arrays) and _is_int_expression(
            node.right, scalars, arrays
        )
    return False


def integer_division_problems() -> list[str]:
    """
    Check 17: an integer ``/`` inside a ``@wp.kernel`` or ``@wp.func`` body.

    ``.claude/CLAUDE.md`` section 1.5: on integers Warp's ``/`` and ``//`` are the *same* operation
    -- both truncate toward zero, unlike CPython's ``//``, which floors. So this is a legibility
    rule and not a correctness one: ``/`` on two ``int32``s reads as real division and truncates
    only because the operands happen to be integers, which means a reader has to recover both types
    before they know what the line does. Every dividend in this package is a non-negative index,
    where the two conventions coincide; the hazard the spelling creates is *porting* such a line
    between host Python and kernel scope, where the answer changes silently for a negative dividend.

    Why a check rather than a one-off edit: the third pass converted eight sites and wrote the rule
    into ``CLAUDE.md``, and the next module written after it reintroduced two
    (``algorithms/multigrid.py``'s ``column = t / n_rows`` and ``column = t / stride``). That is
    section 14's bar for a new check -- the same defect found twice -- and the same failure mode
    check 16 exists to prevent for casts.

    The scan is deliberately conservative, because the cost of a false positive is that the first
    person it annoys disables it. An operand counts as an integer only *by declaration*: an
    annotated ``wp.int*`` / ``wp.uint*`` / ``wp.Int`` parameter, an element of a ``wp.array`` whose
    annotated dtype is one of those, an integer literal, a ``.shape[...]``, ``wp.tid()``, an
    integer constructor, or an integer-preserving expression over those. A ``wp.Scalar``-generic
    parameter, a float, and anything whose type comes from a call the scan cannot see are all left
    untyped, so their divisions are never reported.
    """
    problems: list[str] = []
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue  # test_package_scan_is_discoverable reports the parse failure
        constants = _int_module_constants(tree)
        for function in _kernel_scope_functions(tree):
            scalars, arrays = _int_typed_names(function, constants)
            for node in ast.walk(function):
                if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
                    continue
                if not _is_int_expression(node.left, scalars, arrays):
                    continue
                if not _is_int_expression(node.right, scalars, arrays):
                    continue
                problems.append(
                    f"{path.relative_to(_REPO_ROOT)}:{node.lineno} {function.name} divides "
                    f"integers with '/' in '{ast.unparse(node)}' -- write // instead, which is the "
                    "same operation on integers and says so"
                )
    return problems


# --- check 18 -----------------------------------------------------------------------------------

# Bare Python annotations with a Warp equivalent. ``str`` is deliberately absent: no kernel-scope
# argument can be one, so a ``str`` annotation is proof the function is a Python-scope factory and
# not kernel code at all.
_BARE_ANNOTATIONS = {"bool": "wp.bool", "int": "wp.int32", "float": "wp.float32"}


def _annotation_nodes(function: ast.FunctionDef) -> list[tuple[str, ast.expr]]:
    """Every annotation in a signature, paired with the argument name (``->`` for the return)."""
    arguments = function.args
    every = [
        *arguments.posonlyargs,
        *arguments.args,
        *([arguments.vararg] if arguments.vararg else []),
        *arguments.kwonlyargs,
        *([arguments.kwarg] if arguments.kwarg else []),
    ]
    found = [(argument.arg, argument.annotation) for argument in every if argument.annotation]
    if function.returns is not None:
        found.append(("->", function.returns))
    return found


def bare_annotation_problems() -> list[str]:
    """
    Check 18: a bare ``bool`` / ``int`` / ``float`` annotation in a kernel-scope signature.

    ``.claude/CLAUDE.md`` section 1.2: kernel arguments and returns are spelled in Warp's types.
    Warp resolves the bare names to the same ones, so like checks 16 and 17 this is legibility
    rather than correctness -- and like them, that is exactly why nothing but a scan holds it. The
    tree carried 11 ``-> bool`` against 46 ``-> wp.bool``, five of them predating the fourth kernel
    pass and the newest written the day after it, which is section 14's bar for a check: the same
    defect found twice, in code written after the rule.

    The distinction that makes it sound is that it reads *only* ``@wp.kernel`` / ``@wp.func``
    signatures. A kernel **factory** is ordinary Python and its parameters are correctly plain --
    ``reduce.blocks_1d(n: int) -> int`` and ``neighbors._bvh_nearest_row_kernel(row_size: int,
    name: str)`` compute a launch geometry and a kernel name at import time, and annotating them
    ``wp.int32`` would be a lie about where they run. Those are not kernel-scope functions, so they
    are never scanned; ``str`` is left out of the mapping for the same reason, as a further tell.
    """
    problems: list[str] = []
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue  # test_package_scan_is_discoverable reports the parse failure
        for function in _kernel_scope_functions(tree):
            for name, annotation in _annotation_nodes(function):
                if not isinstance(annotation, ast.Name):
                    continue
                replacement = _BARE_ANNOTATIONS.get(annotation.id)
                if replacement is None:
                    continue
                where = "return" if name == "->" else f"argument '{name}'"
                problems.append(
                    f"{path.relative_to(_REPO_ROOT)}:{annotation.lineno} {function.name} annotates "
                    f"its {where} '{annotation.id}' -- write {replacement} instead"
                )
    return problems


# --- check 19 -----------------------------------------------------------------------------------

# The suffixes ``.claude/CLAUDE.md`` section 7.1 assigns to reference libraries. ``_np`` is
# deliberately absent: section 7.1 gives it to "NumPy/SciPy" and ``tests/parity.py`` counts it,
# which is right there because a ``parity`` marker has already declared that a second implementation
# was consulted -- but in the suite at large ``_np`` marks *inputs* at least as often as oracles.
# Measured: adding it takes this scan from 523 comparison tests to 783 and from 0 problems to 290.
#
# ``_gl`` is moderngl's, and this table has to be edited alongside ``tests/parity.py``'s
# ``_LIBRARY_SUFFIXES`` -- the two encode the same convention independently, so a reference added
# to one and not the other is enforced by half the gate. Measured clean: the only ``*_gl`` names in
# the suite are ``image_gl``, ``covered_gl`` and ``class_gl``, all moderngl's.
_REFERENCE_SUFFIXES = ("_tm", "_igl", "_pp", "_pml", "_o3d", "_pv", "_ml", "_pmf", "_gl", "_p3d")

# The four phrases the suite uses to label a comparison, all four in good standing. ``Class [ABCD]``
# and ``Not a library comparison`` are section 6's named labels; ``Not a parity assert`` and
# ``Triwarp against triwarp`` are its triwarp-against-triwarp family. A gate accepting only the
# first two would fail 14 correct tests and the author's fix would be to reword good docstrings.
_COMPARISON_LABELS = re.compile(
    r"Class [ABCD]\b|Not a library comparison|Not a parity assert|[Tt]riwarp against triwarp"
)


def _asserted_reference_names(function: ast.FunctionDef) -> set[str]:
    """Reference-suffixed names read by an ``assert`` in this function."""
    found: set[str] = set()
    for statement in ast.walk(function):
        if not isinstance(statement, ast.Assert):
            continue
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and node.id.endswith(_REFERENCE_SUFFIXES):
                found.add(node.id)
    return found


def comparison_label_problems() -> list[str]:
    """
    Check 19: a test comparing against a reference library says which class the comparison is.

    ``.claude/CLAUDE.md`` section 7.4 asks for the label and explains what each class means; this
    only checks that one of the four phrases is present. It cannot check that the label is the
    *right* one -- that stays a review question, as section 4.5 says of every naming rule -- and it
    must not try: a ``_tm`` name inside an ``assert`` is not proof of an oracle.
    ``test_split_single_component`` compares ``split``'s output against ``mesh_tm.vertices``, the
    *input* mesh, which is a round trip; and in
    ``test_split_faces_along_field_positive_side_is_the_clip`` the ``_tm`` names are triwarp results
    run through ``warp_to_trimesh``. Both are correctly labelled and neither is a library
    comparison.

    Two decisions make it fire only on real omissions. It keys on ``ast.Assert`` rather than on the
    whole function body, because a fixture unpack ``mesh_tm, mesh_wp = icosphere`` names a ``_tm``
    variable in every mesh test -- keying on the body takes this from 0 problems to 120. And it
    accepts all four label phrases rather than section 6's two headline ones, for the reason given
    at ``_COMPARISON_LABELS``.

    It exists because the convention has decayed twice. Round 2 of the test-suite rescan took the
    lowercase ``class b`` spelling from 21 to 0, and 9 more had accumulated by round 4 -- invisible
    to the grep section 6 prescribes, since a human reads them the same. The other defect it closes
    is rarer and worse: ``test_fill_min_weight_matches_meshlib`` carried a ``parity`` marker,
    compared a class-C statistic against MeshLib, and had **no docstring at all**, which ruff cannot
    see because ``D103`` is in the ignore list.
    """
    problems: list[str] = []
    for path in sorted(_TESTS_DIR.glob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue  # test_package_scan_is_discoverable reports the parse failure
        for function in ast.walk(tree):
            if not isinstance(function, ast.FunctionDef) or not function.name.startswith("test_"):
                continue
            names = _asserted_reference_names(function)
            if not names:
                continue
            docstring = ast.get_docstring(function) or ""
            if _COMPARISON_LABELS.search(docstring):
                continue
            reason = "has no docstring" if not docstring else "carries no class label"
            problems.append(
                f"{path.relative_to(_REPO_ROOT)}:{function.lineno} {function.name} {reason} "
                f"but asserts on {', '.join(sorted(names)[:3])} -- write 'Class A'/'Class B'/"
                f"'Class C'/'Class D', 'Not a library comparison' or 'Not a parity assert'"
            )
    return problems


# --- check 20 -----------------------------------------------------------------------------------


def kernel_scope_ternary_problems() -> list[str]:
    """
    Check 20: a Python ternary (``a if cond else b``) inside a ``@wp.kernel`` or ``@wp.func`` body.

    ``.claude/CLAUDE.md`` section 1.5: the tree's spelling for a conditional value at kernel scope
    is ``wp.where(cond, a, b)``, 33+ sites in 16 kernel modules. A ternary compiles to the same code
    -- Warp lowers ``ast.IfExp`` the same way it lowers a call to ``wp.where`` -- so like checks 16,
    17 and 18 this is legibility rather than correctness, and like them nothing but a scan holds the
    line. The sixth kernels pass declared this axis at zero and was wrong: two ternaries in
    ``kernels/intersection.py`` (``git log -L`` puts them at a line-length reformat, well before
    that pass) survived it and at least two review passes before. Converting bare ``wp.where``
    eagerly evaluates both arms where a ternary would short-circuit; every kernel-scope ternary
    found to date has both arms already evaluated (an argsort output, an index), so the two are
    interchangeable at the sites this check has found -- a future site where they are not needs a
    comment explaining why the ternary stays, not a blanket exemption.
    """
    problems: list[str] = []
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue  # test_package_scan_is_discoverable reports the parse failure
        for function in _kernel_scope_functions(tree):
            for node in ast.walk(function):
                if not isinstance(node, ast.IfExp):
                    continue
                problems.append(
                    f"{path.relative_to(_REPO_ROOT)}:{node.lineno} {function.name} uses a "
                    f"ternary ('{ast.unparse(node)}') -- write wp.where(cond, a, b) instead"
                )
    return problems


# --- check 21 -----------------------------------------------------------------------------------

# Two libraries the shipped package may not cite, in one pattern because the fix is the same for
# both: say what the code computes, or name the algorithm in the literature's vocabulary. Their
# *reasons* differ and are worth keeping apart -- MeshLib first, then promesh below.
#
# MeshLib's licence restricts *use*, not merely distribution of derivatives, and triwarp ships
# ``MIT OR Apache-2.0`` -- so ``.claude/CLAUDE.md``'s MeshLib block requires that nothing under
# ``triwarp/`` name the library at all: not the library, not one of its C++ functions, not one of
# its source files. This is the pattern that sentence's grep asks for, widened in two ways the
# eighth kernels pass measured as necessary. The ``MR`` prefix covers a source-file or class name
# generically (the block's own list of four symbols matched *none* of the three file-name comments
# found in ``kernels/``, because they name ``MRLaplacian.cpp`` and
# ``MRPointCloudTriangulationHelpers.cpp``); the named C++ identifiers cover the two comments that
# said "port of" in the imperative, which is the sharper half of the defect and carries no ``MR``.
# Prose mentions in ``tests/`` and ``benchmarks/`` are correct and required -- a comparison has to
# say what it compares against -- so the scan is scoped to the package.
#
# ``promesh`` is here for a *related but distinct* reason, and the difference is worth stating so
# the next reader does not merge the two rules. It is not proprietary -- it is
# **unverifiable**: ``reference/promesh/`` is a bare source drop with no ``LICENSE``, no
# ``COPYING`` and no ``pyproject.toml``, so its terms cannot be read from this repo at all, and it
# is not a published package (nothing installs it, so it can never be a comparison row either).
# Five references had accumulated in ``holes.py``, two of them claiming a *port* ("the Warp port
# of promesh's ``triangulate_boundaries``", "Mirrors promesh's private helper") -- a derivation
# claim against terms nobody here has checked, which is the same hazard as the MeshLib half even
# though the licence story is the opposite. All five were rewritten into the algorithm's own
# vocabulary (Barequet & Sharir's gap bridging, the minimal-perimeter heuristic, the
# longest-increasing-subsequence monotonicity correction), which is what a reader wanted anyway.
_UNCITABLE_REFERENCES = re.compile(
    # The ``MR`` source-file / class prefix is matched case-**sensitively**, inside a scoped
    # ``(?-i:)``, so that ``IGNORECASE`` on the rest cannot turn it into "any word starting with
    # mr". The ``\b`` is what keeps ``circumradius`` from matching it either way.
    r"\bmeshlib\b|\bmrmesh(py|numpy)\b|(?-i:\bMR[A-Z][A-Za-z]{2,})"
    r"|\bFanOptimizer\b|\bbuildLocalTriangulation\b|\bpositionVertsSmoothly"
    r"|\bcalcQueueElement|\bupdateBorderQueueElement"
    r"|\bpromesh\b|\btriangulate_boundaries\b",
    re.IGNORECASE,
)


def uncitable_reference_problems() -> list[str]:
    """
    Check 21: a name anywhere under ``triwarp/`` that the shipped package may not cite.

    Two libraries qualify, for opposite reasons, and the check is one scan because the *fix* is
    identical either way.

    **MeshLib** -- ``.claude/CLAUDE.md``'s MeshLib block: *"Nothing under ``triwarp/`` may name
    MeshLib at all"*, because its licence restricts use rather than distribution and triwarp ships
    ``MIT OR Apache-2.0``, so an attribution comment collectively reads as a claim that a
    permissively licensed package is derived from a proprietary one. 89 such references were
    removed in one pass across 19 files; five had come back by the eighth kernels pass, two of them
    stating the claim outright ("port of ``FanOptimizer``"). A rule whose enforcement is "someone
    remembers to grep" has now failed once, which is what this check is for.

    **promesh** -- not proprietary but *unverifiable*, and not a published package: its mirror
    carries no licence file of any kind, and nothing installs it. Five references in ``holes.py``
    claimed a port from it. See the comment above the pattern.

    Describe what the code computes, or name the algorithm in the literature's vocabulary -- which
    is what section 10 asks for independently and what a reader needed anyway. Reading either
    mirror to understand an operation's *interface* stays allowed; naming it here does not.
    """
    problems: list[str] = []
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            match = _UNCITABLE_REFERENCES.search(line)
            if match is None:
                continue
            problems.append(
                f"{path.relative_to(_REPO_ROOT)}:{number} names a library the package may not "
                f"cite ('{match.group(0)}') -- say what the code computes, or name the algorithm"
            )
    return problems


# --- check 22 -----------------------------------------------------------------------------------

_TID_CASTS = frozenset({"wp.int32", "wp.int64", "wp.uint32", "wp.uint64"})


def _is_tid_call(node: ast.expr) -> bool:
    """Report whether this expression is exactly ``wp.tid()``, with no arguments."""
    return (
        isinstance(node, ast.Call)
        and not node.args
        and not node.keywords
        and _dotted(node.func) == ("wp", "tid")
    )


def bare_tid_problems() -> list[str]:
    """
    Check 22: a single-index ``wp.tid()`` assigned without the declarative ``wp.int32`` cast.

    ``.claude/CLAUDE.md`` section 1.3 retires every redundant cast and keeps exactly one -- the tid
    cast -- *because* it is the declarative one: it names the type of the index the whole kernel is
    written against. ``wp.tid()`` already returns ``wp.int32``, so both spellings generate
    identical code and neither the compiler nor the suite can see the difference; this is the fifth
    member of the family checks 16, 17, 18 and 20 belong to, and like them nothing but a scan holds
    the line. When it was written ``kernels/`` was about 90 % cast and 10 % bare, and the drift was
    *per file* rather than scattered -- which is the signature of a convention that was never
    checked. ``intersection.py`` was the sharpest case: ``emit_quad_cut`` and ``emit_tri_cut``
    opened bare while ``emit_split_cut_edges`` and ``emit_split_cut_corner``, four kernels emitting
    the same family of triangles in one file, opened with the cast.

    A **multi-index** unpack (``i, j = wp.tid()``) cannot carry a cast and is out of scope by
    construction, which is the one thing that would make this misfire -- 61 such sites are correct
    as they are and the check never looks at a tuple target.
    """
    problems: list[str] = []
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue  # test_package_scan_is_discoverable reports the parse failure
        for function in _kernel_scope_functions(tree):
            for node in ast.walk(function):
                if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                    continue
                if not isinstance(node.targets[0], ast.Name):
                    continue  # a tuple target is a multi-index unpack: out of scope
                if not _is_tid_call(node.value):
                    continue
                name = node.targets[0].id
                problems.append(
                    f"{path.relative_to(_REPO_ROOT)}:{node.lineno} {function.name} opens "
                    f"'{name} = wp.tid()' -- write '{name} = wp.int32(wp.tid())', the one cast "
                    f"the tree keeps because it is declarative"
                )
    return problems


# --- check 23 -----------------------------------------------------------------------------------

# Modules whose repeated ``wp.map`` sites are measured *not* to fork: every one of them reaches a
# single call signature, so a declaration table would be import cost for nothing. Derived from the
# same census as the tables themselves -- add an entry only with that measurement, never to quiet
# the check, because the whole point of the census is that a fork is invisible without it.
_MAP_DECLARATION_ALLOWLIST: frozenset[str] = frozenset(
    {
        # ``reciprocal_or_zero`` from three sites, all float32.
        "energies",
        # ``abs_deviation`` from two sites, both float32.
        "registration",
        # ``extract_components`` from three and ``add_scaled_normal`` / ``combine_components``
        # from two each -- every one of them a single signature, because the smoothers are float32
        # throughout and their repeated sites differ in the *buffer*, not in its dtype, rank or
        # length. (``laplacian_step`` was a fourth until the explicit filters fused their operator
        # apply into the step kernel and stopped mapping it at all.)
        "smoothing",
    }
)


def map_declaration_problems() -> list[str]:
    """
    Check 23: a kernel module that maps its own ``@wp.func`` at two dtypes with no declaration.

    ``.claude/CLAUDE.md`` section 2.5's ``_register_overloads`` rule, one construct over. ``wp.map``
    generates a module named ``map_<unqualified op name>`` and each distinct *call signature* forks
    its hash, so reaching one op at three signatures builds its module three times -- dozens of
    redundant cold builds over one suite run. ``kernels/array.py``'s ``declare_map_signatures``
    block carries the whole reasoning and the fork axes.

    This is the same *kind* of check as ``test_generic_kernels_register_their_overloads``: it
    asserts only that a module which needs a declaration table **has** one, never that the table
    is complete -- proving that means launching the whole dispatch, and the cost of getting it
    wrong is a rebuild rather than a wrong answer. The completeness gate is the load census the
    block above documents, which is a clock measurement and not an assert.

    A module is flagged when the *wrappers* map one of its ``@wp.func``s from two or more distinct
    call sites and it has no ``_declare_map_kernels``. Two sites is the trigger rather than two
    dtypes because a static scan cannot see a dtype: what it can see is that the same op is mapped
    from more than one place, which is the precondition for a fork.
    """
    declared = {
        path.stem
        for path in _KERNELS_DIR.rglob("*.py")
        if "_declare_map_kernels" in path.read_text(encoding="utf-8")
    }
    # ``wp.map(kernel_<module>.<op>, ...)`` at Python scope, per (module, op).
    call = re.compile(r"wp\.map\(\s*kernel_(\w+)\.(\w+)")
    sites: dict[tuple[str, str], int] = {}
    for path in sorted(_PACKAGE_DIR.glob("*.py")):
        for module, op in call.findall(path.read_text(encoding="utf-8")):
            sites[(module, op)] = sites.get((module, op), 0) + 1
    problems: list[str] = []
    for (module, op), count in sorted(sites.items()):
        if count >= 2 and module not in declared and module not in _MAP_DECLARATION_ALLOWLIST:
            problems.append(
                f"triwarp/kernels/{module}.py maps '{op}' from {count} call sites and has no "
                f"_declare_map_kernels() -- see kernels/array.py::declare_map_signatures"
            )
    return problems


# --- check 24 -----------------------------------------------------------------------------------

# A cross-reference into ``.claude/CLAUDE.md``, in every spelling the tree uses: either file name
# -- ``AGENTS.md`` is a symlink to it and three comments cite it that way -- optionally in single
# or double backticks and optionally with a ``.claude/`` prefix, then either the word ``section``
# and a number or a bare ``§`` and a number. ``CLAUDE.md 13.1`` (no connecting word) is a further
# spelling and is matched too, because it is the same claim with the word dropped.
#
# What this cannot see is a reference that names the file in one sentence and the number in the
# next ("``.claude/CLAUDE.md`` ... which is section 3's point about"). Those exist and were fixed
# by hand; a regex that matched a bare "section N" anywhere would also match ``kernels/remesh.py``
# citing "Liepa 2003, section 3", which is a paper and not this file.
_CLAUDE_SECTION_REFERENCE = re.compile(
    r"(?:\.claude/)?(?:``|`)?(?:CLAUDE|AGENTS)\.md(?:``|`)?(?:'s)?\s*"
    r"(?:section\s+|§\s*)?(\d+)(?:\.(\d+))?\b"
)

# The file itself, and the headings it is the authority for. ``## 7. Testing`` is a chapter,
# ``### 7.4 The parity gate`` a section.
_CLAUDE_MD = _REPO_ROOT / ".claude" / "CLAUDE.md"
_CLAUDE_HEADING = re.compile(r"^#{2,3}\s+(\d+)(?:\.(\d+))?[.\s]", re.MULTILINE)

# Chapter-level references that are deliberately a whole chapter although that chapter *is*
# subdivided, because the sentence is about its subject as a whole and no single subsection owns
# it. Keyed by ``(module, "N")`` the way check 9's allowlist is keyed. A chapter with no
# subsections needs no entry -- the bare number is the only spelling available and the check reads
# the heading structure rather than a list.
_CLAUDE_CHAPTER_ALLOWLIST: dict[tuple[str, str], str] = {}


def claude_section_reference_problems() -> list[str]:
    """
    Check 24: a ``.claude/CLAUDE.md`` cross-reference naming a section that does not exist.

    Two failure modes, and the second is the one that motivated the check. A reference to a
    subsection that is not a heading in the file is simply broken. A reference to a bare *chapter*
    number is rejected **when that chapter is subdivided**, even though it resolves, because a
    chapter number is what let 111 of these rot undetected: CLAUDE.md's Part I was reorganised and a
    comment citing "section 4" for the ``wp.map`` rule kept resolving -- to "Evolving the public
    API", which says nothing about ``wp.map``. One number standing for several unrelated sections
    is exactly what a resolving check cannot see, so where a subsection exists it is the
    convention.

    **The subdivision test is what keeps this from being a rule half its sites would have to be
    allowlisted out of.** Six chapters (5, 6, 8, 9, 10, 11) carry no ``###`` heading at all, so a
    bare number is the only spelling available for them and flagging it would be flagging the
    correct citation. Reading that from the heading structure rather than from a written list also
    means the check follows CLAUDE.md if a chapter later gains subsections.

    Scans the same three roots check 9 does. Abstains when ``.claude/CLAUDE.md`` is absent -- it
    does not ship in the wheel, and a check that cannot read its authority reports nothing rather
    than guessing.
    """
    if not _CLAUDE_MD.is_file():
        return []
    text = _CLAUDE_MD.read_text(encoding="utf-8")
    chapters = {m.group(1) for m in _CLAUDE_HEADING.finditer(text) if m.group(2) is None}
    sections = {f"{m.group(1)}.{m.group(2)}" for m in _CLAUDE_HEADING.finditer(text) if m.group(2)}
    subdivided = {section.split(".")[0] for section in sections}

    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    for root, prefix in _WARP_VERSION_SCAN_ROOTS:
        if not root.is_dir():  # neither suite ships in the wheel
            continue
        for path in sorted(root.rglob("*.py")):
            module = path.relative_to(root).with_suffix("").as_posix().replace("/", ".")
            module = prefix + module.removesuffix(".__init__").lstrip(".")
            site = path.relative_to(_REPO_ROOT)
            for lineno, block in _prose_blocks(path.read_text(encoding="utf-8")):
                for match in _CLAUDE_SECTION_REFERENCE.finditer(block):
                    line = lineno + block.count("\n", 0, match.start())
                    chapter, sub = match.group(1), match.group(2)
                    if sub is not None:
                        if f"{chapter}.{sub}" not in sections:
                            problems.append(
                                f"{site}:{line}: cites CLAUDE.md section {chapter}.{sub}, which "
                                "is not a heading in .claude/CLAUDE.md -- the file was "
                                "reorganised, so re-read the sentence and cite the section it "
                                "now means"
                            )
                        continue
                    key = (module, chapter)
                    if key in _CLAUDE_CHAPTER_ALLOWLIST:
                        seen.add(key)
                        continue
                    if chapter in chapters and chapter not in subdivided:
                        continue  # no subsection exists, so the bare number is the citation
                    detail = (
                        f"chapter {chapter} is subdivided, so a bare number under-specifies it"
                        if chapter in chapters
                        else f"chapter {chapter} is not a heading at all"
                    )
                    problems.append(
                        f"{site}:{line}: cites CLAUDE.md section {chapter} -- {detail}. Cite the "
                        "subsection (N.M) the sentence actually means, or add a "
                        "_CLAUDE_CHAPTER_ALLOWLIST entry arguing the whole chapter is the target"
                    )
    problems.extend(
        f"_CLAUDE_CHAPTER_ALLOWLIST entry {key!r} matches nothing in the scanned tree -- drop it"
        for key in sorted(_CLAUDE_CHAPTER_ALLOWLIST)
        if key not in seen
    )
    return problems


# --- check 25 -----------------------------------------------------------------------------------

# The numpydoc sections that are *item lists*: a header, a rule of dashes, then one entry per line
# with its description indented under it. An admonition between two entries of one of these is the
# defect; ``Notes``, ``Examples``, ``Warnings`` and the leading description are free prose and are
# where an admonition belongs.
_NUMPYDOC_ITEM_SECTIONS = frozenset(
    {"Parameters", "Returns", "Yields", "Receives", "Raises", "Warns", "Attributes", "See Also"}
)
_SECTION_RULE = re.compile(r"^-{3,}$")


def admonition_placement_problems() -> list[str]:
    """
    Check 25: a ``!!!`` admonition inside a numpydoc *item-list* section.

    griffe parses a numpydoc section by reading each entry's first line as a name -- a parameter
    name, an exception type -- so an ``!!! note "..."`` header sitting between two entries of a
    ``Raises`` block becomes an **exception type** on the rendered API page, and the admonition's
    own body is swallowed as that exception's description. The page grows a row for a type that
    does not exist and the warning the author wrote never renders as one. Nothing else sees it:
    ``zensical build --strict`` is clean, because every cross-reference in the swallowed text still
    resolves, and ruff's ``D`` rules do not model section contents.

    It had decayed to **eight** sites across six modules before anyone looked, six of them in the
    ``Raises`` block of a function whose caveat is genuinely worth reading -- which is the tell
    that this is a convention people get wrong rather than a rule they flout: an admonition is
    written where the thought occurs, and the thought occurs while documenting what the function
    rejects.

    **It ships with no allowlist, deliberately.** Unlike the section-number check above there is no
    legitimate instance to exempt: an item-list section is a list of items, and every admonition
    has a correct home one section down (``Notes``) or in the leading description, with no loss of
    meaning and no reordering of anything a caller reads first. An allowlist here would only ever
    hold a site nobody had moved yet.

    Scans every module under ``triwarp/``, ``kernels/`` included. Nothing in ``kernels/`` renders,
    so a hit there is cosmetic rather than a broken page -- but the convention is the same one, the
    scan is the same scan, and a rule that holds in one half of the tree and not the other is a
    rule the next reader has to look up.

    **What it deliberately does not catch is the general case, which is any free prose between two
    entries** -- a plain sentence appended to a ``Returns`` block becomes a third, *nameless*
    return, confirmed the same way. Only the ``!!!`` form is scanned, because separating stray
    prose from an entry's own indented continuation needs the indentation rules numpydoc itself is
    loose about, and a check that guesses there would misfire on correct docstrings. The admonition
    form is the one that decayed, and it is unambiguous.
    """
    problems: list[str] = []
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        site = path.relative_to(_REPO_ROOT)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue  # check 12 and the suite itself report an unparseable module
        for node in ast.walk(tree):
            if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef):
                continue
            docstring = ast.get_docstring(node, clean=False)
            if docstring is None or "!!!" not in docstring:
                continue
            first_line = 1 if isinstance(node, ast.Module) else node.body[0].lineno
            lines = docstring.splitlines()
            section: str | None = None
            for index, line in enumerate(lines):
                stripped = line.strip()
                if _SECTION_RULE.fullmatch(stripped) and index:
                    section = lines[index - 1].strip()
                    continue
                if stripped.startswith("!!!") and section in _NUMPYDOC_ITEM_SECTIONS:
                    name = getattr(node, "name", "<module>")
                    problems.append(
                        f"{site}:{first_line + index}: {name}'s {section!r} block holds an "
                        f"admonition ({stripped.split(chr(34))[0].strip()}) -- griffe reads its "
                        "header as an entry name, so the page grows a bogus row and the "
                        "admonition never renders. Move it to Notes or to the description"
                    )
    return problems


# --- check 26 -----------------------------------------------------------------------------------

# Warp's scalar, vector and matrix constructors. A value built by one of these is a
# ``warp._src.types`` instance, *not* a Python number, and its operators route through Warp's
# Python-scope builtin dispatch. ``wp.constant`` is deliberately absent: on Warp 1.17 it is
# ``return x`` after a validity check, so ``wp.constant(7)`` is a plain ``int`` and only
# ``wp.constant(wp.int32(7))`` is Warp-typed -- the check unwraps it and looks at what is inside.
_WARP_TYPED_CONSTRUCTORS: frozenset[str] = frozenset(
    {f"{kind}{bits}" for kind in ("int", "uint", "float") for bits in (8, 16, 32, 64)}
    | {f"vec{n}{suffix}" for n in (2, 3, 4) for suffix in ("", "b", "h", "i", "l", "f", "d")}
    | {f"mat{n}{n}{suffix}" for n in (2, 3, 4) for suffix in ("", "f", "d")}
    | {"quat", "quatf", "quatd", "transform", "transformf", "transformd", "spatial_vector"}
)

# Nothing legitimate does host arithmetic on a Warp-typed constant, so this ships empty. Before
# adding an entry, check the two spellings that are *not* defects and need no exemption: passing
# the constant straight into ``wp.launch(inputs=[...])`` / ``wp.map(...)`` / ``fill_(...)`` as a
# kernel scalar, and ``int(CONST)`` / ``float(CONST)``, which unwraps it for nothing.
_WARP_HOST_ARITHMETIC_ALLOWLIST: dict[tuple[str, str], str] = {}


def _warp_typed_constant_names(tree: ast.Module) -> set[str]:
    """Module-level names bound to a Warp *typed* constructor, unwrapping ``wp.constant``."""
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = node.value
        # ``wp.constant(wp.int32(0))`` -- look through the wrapper at its payload.
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "constant"
            and value.args
        ):
            value = value.args[0]
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr in _WARP_TYPED_CONSTRUCTORS
        ):
            names.add(target.id)
    return names


def _warp_typed_use(
    node: ast.expr, local: set[str], aliases: dict[str, str], constants: dict[str, set[str]]
) -> str | None:
    """Name the Warp-typed constant ``node`` refers to, bare or through a module alias."""
    if isinstance(node, ast.Name) and node.id in local:
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        origin = aliases.get(node.value.id)
        if origin and node.attr in constants.get(origin, set()):
            return f"{node.value.id}.{node.attr}"
    return None


def warp_host_arithmetic_problems() -> list[str]:
    """
    Check 26: a Warp-typed constant used in Python-scope arithmetic or as a slice bound.

    A ``wp.int32`` / ``wp.float32`` / ``wp.vec3`` instance is not a Python number. Its ``__add__``
    and friends are ``warp._src.types.scalar_base``'s, which call ``warp.add(self, y)`` -- Warp's
    Python-scope builtin dispatch, an ``inspect.signature().bind()`` per operand -- **two to three
    orders of magnitude a Python float's operator**. A ``wp.array`` slice taken with such bounds is
    worse still, because ``wp.array.__getitem__`` forms ``stop - start`` and ``strides * start``
    internally, so one Warp-typed bound is three dispatches rather than one.

    The defect is silent, which is why it needs a scan: the answer is correct, the compiler sees
    nothing, and the suite sees nothing. Its siblings are not -- ``//`` and ``%`` on a Warp scalar
    raise ``TypeError``, and ``wp.zeros(wp.int32(n))`` raises too -- so arithmetic and slicing are
    the whole of the hazard.

    **Scope is deliberately narrow.** Uses are read in the wrapper layer (``triwarp/*.py``, which
    holds no kernel bodies at all) and at *module scope* in ``kernels/``; a kernel or ``@wp.func``
    body is where these constants are supposed to be used and is never read. The check also stops
    at constants: extending it to ``wp.length`` / ``wp.cross`` calls would flag mostly legitimate
    sites, which is how a check gets switched off.
    """
    problems: list[str] = []
    constants: dict[str, set[str]] = {}
    trees: dict[str, tuple[Path, ast.Module]] = {}
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue  # check 12 and the suite itself report an unparseable module
        module = path.stem
        constants[module] = _warp_typed_constant_names(tree)
        trees[str(path)] = (path, tree)

    for key, (path, tree) in trees.items():
        del key
        in_kernels = path.parent != _PACKAGE_DIR and "kernels" in path.parts
        local = set(constants.get(path.stem, set()))
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                origin = node.module.rsplit(".", 1)[-1]
                for alias in node.names:
                    if alias.name in constants.get(origin, set()):
                        local.add(alias.asname or alias.name)
                    elif alias.name in constants:
                        aliases[alias.asname or alias.name] = alias.name

        # In ``kernels/`` only module-scope statements are host code; everything inside a function
        # there is a kernel body, a ``@wp.func``, or a factory that builds one.
        roots: list[ast.AST] = list(tree.body) if in_kernels else [tree]
        for root in roots:
            if in_kernels and isinstance(root, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for node in ast.walk(root):
                operands: list[ast.expr] = []
                if isinstance(node, ast.BinOp):
                    operands = [node.left, node.right]
                elif isinstance(node, ast.Slice):
                    operands = [b for b in (node.lower, node.upper, node.step) if b is not None]
                else:
                    continue
                for operand in operands:
                    name = _warp_typed_use(operand, local, aliases, constants)
                    if name is None:
                        continue
                    site = path.relative_to(_REPO_ROOT)
                    if (str(site), name) in _WARP_HOST_ARITHMETIC_ALLOWLIST:
                        continue
                    kind = "slice bound" if isinstance(node, ast.Slice) else "arithmetic operand"
                    problems.append(
                        f"{site}:{node.lineno}: {name} is a Warp-typed constant used as a "
                        f"Python-scope {kind} -- that routes through Warp's builtin dispatch "
                        f"(~10 us per op, 39.4 us for a slice against 3.16). Unwrap it with "
                        f"int(...), or derive a plain module-level value the way "
                        f"kernels/array.py's LOOP_CONDITION_VIEW does"
                    )
    return problems


_TRIPLET_BUILDERS = frozenset({"bsr_from_triplets", "bsr_set_from_triplets"})


def triplet_build_problems() -> list[str]:
    """
    Check 27: a call to ``warp.sparse``'s triplet builders anywhere under ``triwarp/``.

    ``bsr_from_triplets`` sorts every triplet on a full-width key and allocates scratch several
    times the matrix; on a 28 M-face mesh ``cotmatrix`` spent 9.4 GB a call there for a matrix
    under 1 GB, and a drained CUDA mempool (CLAUDE.md 13.1) makes every one of those bytes a cost.
    The package builds its matrices from row-major keys instead -- ``array.csr_from_keys`` when
    the sparsity follows from the mesh, ``array.csr_from_triplets`` when the input is genuinely an
    unordered coordinate list -- so a new call is a regression to route through one of those.
    Tests may still call it: there it builds an *input*, independently of the code under test.

    No allowlist: every former site converted, including the trivial ones (an empty matrix and a
    one-entry-per-row prolongator are written as CSR directly).
    """
    problems: list[str] = []
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        site = path.relative_to(_REPO_ROOT)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue  # check 12 and the suite itself report an unparseable module
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in _TRIPLET_BUILDERS:
                problems.append(
                    f"{site}:{node.lineno}: calls {name} -- assemble the CSR from keys "
                    "(array.csr_from_keys) or through array.csr_from_triplets"
                )
    return problems


# --- check 28 -----------------------------------------------------------------------------------

_DOCS_NAV = _REPO_ROOT / "docs" / "SUMMARY.md"
_DOCS_API_LINK = re.compile(r"\(api/triwarp/(\w+)\.md\)")


def docs_nav_problems() -> list[str]:
    """
    Check 28: a public module missing from ``docs/SUMMARY.md``, listed twice, or no longer there.

    ``api-autonav`` generates one API page per public module and the committed nav shelves them
    by theme. The plugin places no page the nav does not name, so without this a new module ships
    a page nothing links to. Abstains when the nav file is absent (an sdist without ``docs/``).
    """
    if not _DOCS_NAV.exists():
        return []
    listed = _DOCS_API_LINK.findall(_DOCS_NAV.read_text(encoding="utf-8"))
    public = {path.stem for path in _PACKAGE_DIR.glob("*.py") if not path.stem.startswith("_")}
    problems = [
        f"docs/SUMMARY.md: triwarp.{name} is not in the nav -- shelve it in an API section"
        for name in sorted(public - set(listed))
    ]
    problems += [
        f"docs/SUMMARY.md: triwarp.{name} is listed but the module does not exist"
        for name in sorted(set(listed) - public)
    ]
    problems += [
        f"docs/SUMMARY.md: triwarp.{name} is listed {listed.count(name)} times -- once only"
        for name in sorted({name for name in listed if listed.count(name) > 1})
    ]
    return problems


# --- check 29 -----------------------------------------------------------------------------------

# The spellings a shape-first entry replaced. Each opens an entry's description (or a property's
# summary) with words before the shape, which is what the rule exists to stop: four ways of saying
# one thing made a reader parse each entry anew, and the shape is the part read first.
_SHAPE_LATE_OPENINGS = re.compile(r"^(?:Length[- ]``|Shape ``\(|Flat\b|Rank-[1-4] ``\()")
_SHAPE_ITEM_SECTIONS = frozenset(
    {"Parameters", "Other Parameters", "Returns", "Yields", "Attributes"}
)


def _item_openings(docstring: str) -> list[tuple[int, str, str, str]]:
    """``(line index, section, item, first description line)`` of every numpydoc item entry."""
    lines = docstring.splitlines()
    openings: list[tuple[int, str, str, str]] = []
    section: str | None = None
    section_indent = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if _SECTION_RULE.fullmatch(stripped) and index:
            section = lines[index - 1].strip()
            section_indent = len(lines[index - 1]) - len(lines[index - 1].lstrip())
            index += 1
            continue
        indent = len(line) - len(line.lstrip())
        if section in _SHAPE_ITEM_SECTIONS and stripped and indent == section_indent:
            following = index + 1
            while following < len(lines) and not lines[following].strip():
                following += 1
            if following < len(lines):
                description = lines[following]
                if len(description) - len(description.lstrip()) > section_indent:
                    openings.append((following, section, stripped, description.strip()))
        index += 1
    return openings


def shape_spelling_problems() -> list[str]:
    """
    Check 29: an array entry whose description does not open with its shape.

    Every public ``Parameters`` / ``Returns`` / ``Yields`` / ``Attributes`` entry of an array
    writes the shape first, as a code span, and then what the array means: ``(3 * n_faces,)`` flat
    triangle index buffer. Before the rule the package spelled one idea four ways (a leading
    shape span, ``Length-``, ``Shape ``(``, ``Flat ...``) across some 650 entries, so a reader
    re-parsed every entry and a shape was sometimes buried mid-sentence. The scan reads the first
    description line of each entry, and the one-line summary of every property (the ``Trimesh``
    accessors render as attributes, whose summary is their whole entry).

    What it cannot see is the rest of the rule: a shape buried *later* in a sentence that opens
    with something else (``Smoothed ``(n_vertices,)`` positions``), a redundant dtype, an
    abbreviated size name. Telling an array's own shape from a shape the prose merely mentions
    ("the cell ``(0, 0, 0)``") is a reading, so those stay with review; the openings it bans are
    the spellings that decayed, and each is unambiguous.
    """
    problems: list[str] = []
    for path in sorted(_PACKAGE_DIR.glob("*.py")):
        if path.stem.startswith("_"):
            continue
        site = path.relative_to(_REPO_ROOT)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef | ast.FunctionDef) or node.name.startswith("_"):
                continue
            docstring = ast.get_docstring(node, clean=False)
            if docstring is None:
                continue
            first_line = node.body[0].lineno
            if isinstance(node, ast.FunctionDef) and any(
                "property" in ast.unparse(decorator).lower() for decorator in node.decorator_list
            ):
                summary = docstring.strip().splitlines()[0]
                if _SHAPE_LATE_OPENINGS.match(summary):
                    problems.append(
                        f"{site}:{first_line}: property {node.name}'s summary opens with "
                        f"{summary.split()[0]!r} -- write its shape first, as a code span"
                    )
            for index, section, item, description in _item_openings(docstring):
                if _SHAPE_LATE_OPENINGS.match(description):
                    problems.append(
                        f"{site}:{first_line + index}: {node.name}'s {section} entry {item!r} "
                        f"opens with {description.split()[0]!r} -- write its shape first, as a "
                        "code span, e.g. ``(3 * n_faces,)`` flat triangle index buffer"
                    )
    return problems
