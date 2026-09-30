"""
The convention gate: the public API's names, summaries and file layout, checked statically.

One test per convention, so the failing test's *name* says which one was broken. The scan and the
reasoning behind each rule live in [`tests/api_conventions.py`](api_conventions.py); this file is
only the pytest surface -- with two exceptions, both of which exist because a static read cannot
see what they check: the docstring-example test, which executes the examples, and the lazy-package
tests at the end, which import `triwarp` in a *subprocess* because the in-process import graph is
already whatever the session made it.

Deliberately not parametrized over modules or functions: that would add hundreds of always-green
items to every run, and a rule that stopped matching anything would silently lose its check instead
of failing. The example test *is* parametrized, because there are four of them and the failure has
to name which one.
"""

# This file tests the scan helpers of tests/api_conventions.py, private ones included, directly.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from collections.abc import Mapping

import numpy as np
import pytest
import trimesh as tm
import warp as wp

import triwarp as tw
from tests.api_conventions import (
    _UNCITABLE_REFERENCES,
    DocstringExample,
    _annotation_nodes,
    _asserted_reference_names,
    _int_module_constants,
    _int_typed_names,
    _is_int_expression,
    _is_tid_call,
    _kernel_scope_functions,
    admonition_placement_problems,
    allocation_device_problems,
    array_annotation_style_problems,
    bare_annotation_problems,
    bare_tid_problems,
    builtin_cast_problems,
    claude_section_reference_problems,
    comparison_label_problems,
    coverage_location_problems,
    docs_nav_problems,
    docstring_examples,
    duplicate_name_problems,
    helper_order_problems,
    installed_warp_version,
    integer_division_problems,
    kernel_module_problems,
    kernel_output_naming_problems,
    kernel_scope_ternary_problems,
    launch_device_problems,
    library_in_summary_problems,
    map_declaration_problems,
    mask_return_problems,
    private_import_problems,
    scan_package,
    triplet_build_problems,
    uncitable_reference_problems,
    undocumented_raise_problems,
    warp_host_arithmetic_problems,
    warp_suffix_problems,
    warp_version_problems,
)
from tests.conversions import warp_empty
from triwarp.kernels import array as kernel_array


def _fail(headline: str, problems: list[str]) -> None:
    """Fail with a counted headline and one indented line per problem, without a traceback."""
    if not problems:
        return
    body = "\n".join(f"  {problem}" for problem in sorted(problems))
    # pytrace=False: the traceback runs through the scanner and says nothing useful, and pytest
    # truncates long assertion explanations in non-verbose runs -- which is exactly when the full
    # list is wanted.
    pytest.fail(f"{len(problems)} {headline}\n{body}", pytrace=False)


def test_package_scan_is_discoverable() -> None:
    """
    Guard the scan itself: a silently empty scan would make all seven checks vacuously green.

    ``triwarp/`` ships in the wheel, so unlike the parity gate there is no packaged-tree case where
    it can legitimately be missing.
    """
    scan = scan_package()
    _fail("module(s) could not be parsed:", scan.errors)
    assert scan.modules, "triwarp/ has modules but the scan found none"
    assert scan.functions, "triwarp/ has public functions but the scan found none"


def test_summaries_do_not_name_a_reference_library() -> None:
    """
    A one-line summary says what the function returns, not which C++ call it wraps.

    mkdocstrings renders the summary as the entry in the module's API index, so a library name there
    turns the index into a table of bindings -- the same defect ``.claude/CLAUDE.md`` section 4.1
    already forbids one level in, where it rules that a function is named after what it returns.
    Attribution is wanted and stays: one line down, in ``Notes`` or ``See Also``.
    """
    _fail("public summary line(s) naming a reference library:", library_in_summary_problems())


def test_mask_producers_return_boolean_arrays() -> None:
    """
    A ``*_mask`` producer returns ``wp.array[wp.bool]``.

    The suffix is a family, and a member returning something else -- ``filter_unsharp_mask``
    returned vertex *positions* -- makes the whole family unreadable to anyone scanning it. A
    function that consumes a mask is named for its input and is exempt.
    """
    _fail("*_mask function(s) that do not return a boolean array:", mask_return_problems())


def test_module_summaries_do_not_advertise_warp() -> None:
    """A module summary does not end in "(Warp)" or "on NVIDIA Warp": the whole package is Warp."""
    _fail("module summary(ies) carrying a redundant Warp suffix:", warp_suffix_problems())


def test_coverage_lives_beside_its_module() -> None:
    """
    Every module has a ``tests/`` and a ``benchmarks/`` file named for it, and vice versa.

    ``.claude/CLAUDE.md`` section 4.2's "coverage is per module" rule, made mechanical. The failure
    it catches is a function's tests drifting into a neighbour's file, which is invisible until
    somebody goes looking for them -- and a suite file whose module has been renamed out from under
    it.
    """
    _fail("coverage file(s) not named for their module:", coverage_location_problems())


def test_private_names_stay_in_their_module() -> None:
    """
    A private helper with callers in two modules is a public function that was written in a hurry.

    ``proximity._default_mesh_query_max_dist`` had four importers, one of which aliased it straight
    back to a public-looking name on import -- the same fact stated as a workaround. The allowlist
    in the scanner carries a written reason per entry.
    """
    _fail("cross-module private import(s):", private_import_problems())


def test_public_names_are_unique_across_modules() -> None:
    """
    Two modules do not export the same public name, outside a written allowlist.

    ``points.centroid`` and ``triangles.centroid`` differed in what they meant (point mean against
    area-weighted surface centroid) *and* in whether calling them synced the device. The allowlist
    holds ``concatenate``, where both spellings are required by the vocabularies they mirror.
    """
    _fail("duplicated public name(s):", duplicate_name_problems())


def test_kernel_modules_are_named_for_their_wrapper() -> None:
    """
    ``triwarp/kernels/<module>.py`` backs ``triwarp/<module>.py``, one to one.

    ``.claude/CLAUDE.md`` section 3.1's rule. It is what stops a wrapper module from being created
    or renamed while its kernels are left behind under the old name -- the half of a move that
    compiles fine and is therefore easy to skip. ``predicates`` and ``scatter`` are the shared
    kernel-side libraries that back no single module; sub-packages mirror a folder and are not
    checked here.
    """
    _fail("kernel/wrapper module name mismatch(es):", kernel_module_problems())


def test_private_helpers_follow_their_callers() -> None:
    """
    A private helper is defined below the public function that calls it (the stepdown rule).

    ``.claude/CLAUDE.md`` section 5: a reader should never need to jump backward to a definition
    they have not been introduced to yet. ``_HELPER_ORDER_ALLOWLIST`` carried the 49 sites that
    predated this check as an explicit debt list rather than a silent exemption, and the staleness
    half of it did its job: the list is now drained to a single permanent entry, a helper called at
    module scope to build a constant, which has no caller to sit below.
    """
    _fail("private helper(s) above their first caller:", helper_order_problems())


def test_warp_version_claims_are_not_stale() -> None:
    """
    No comment or docstring blames a Warp version older than the installed ``warp-lang``.

    The defect this exists for: the 1.16 upgrade re-stamped all five local Warp API mirrors --
    their version stamps make that checkable -- and left twelve *code* justifications citing
    bugs in 1.13-1.15, none re-probed. Six of those bugs were still real and one was not,
    and nothing in the tree could tell them apart. Unlike the other eight checks this one scans
    ``kernels/`` too, since five of the twelve lived there.

    ``_WARP_VERSION_ALLOWLIST`` is for a claim that deliberately records history, and the entry
    carries the reason -- so the next upgrade inherits a list of claims to re-run.
    """
    if installed_warp_version() is None:
        pytest.skip("warp-lang is not installed, so there is no version to compare against")
    _fail("stale Warp-version claim(s):", warp_version_problems())


def test_allocations_name_their_device() -> None:
    """
    Every Python-scope ``wp.zeros``/``empty``/``ones``/``full``/``array`` names a ``device``.

    Without it the buffer lands on Warp's *current* device rather than the device of the arrays it
    is about to be used with, and the suite cannot see the difference: a test runs with its arrays'
    device current, so the omitted argument resolves correctly by accident.
    ``array.index_sparse`` raised only under ``wp.ScopedDevice("cpu")`` with ``cuda:0`` inputs.
    """
    _fail("allocation(s) without device=:", allocation_device_problems())


def test_launches_name_their_device() -> None:
    """
    Every ``wp.launch`` / ``wp.launch_tiled`` names the ``device`` it launches on.

    An omitted one resolves to Warp's current device -- ``cuda:0`` whenever CUDA is present -- so
    with CPU arrays the kernel runs on the GPU over host pointers. Warp permits that by design
    where the GPU can address host memory, and it returns the *correct answer*, which is why
    ``vertices.mean_vertex_normals`` shipped it: the launch acquires no ordering, the host buffers
    are freed at scope exit while the kernel still reads them, and the process aborts later inside
    glibc (20/20 aborts with a free and no synchronize, 0/20 with either).

    The ``STRICT`` launch mode set in ``tests/conftest.py`` is the runtime half of this guard, and
    it cannot replace the scan: it only fires when the arrays are not on the launch device, so on a
    CUDA run the default device is the arrays' device and the omission is invisible to it.
    """
    _fail("launch(es) without device=:", launch_device_problems())


def test_strict_launch_mode_rejects_a_cross_device_launch() -> None:
    """
    The ``STRICT`` mode set in ``tests/conftest.py`` is in force and actually raises.

    Setting a config flag proves nothing on its own: ``launch_array_access_mode`` is consulted per
    launch, and a suite whose arrays all sit on the launch device would pass identically with the
    flag unset. So this deliberately mismatches one launch and requires the exception. ``CHECKED``
    would *not* raise here -- it validates addressability, which HMM provides -- which is why the
    harness sets ``STRICT``.
    """
    assert wp.config.launch_array_access_mode == wp.config.LaunchArrayAccessMode.STRICT
    if not wp.is_cuda_available():
        pytest.skip("a cross-device launch needs a CUDA device to launch on")
    indices = warp_empty(4, wp.int32, "cpu")
    with pytest.raises(RuntimeError, match="device"):
        wp.launch(kernel_array.arange, dim=4, inputs=[indices], device="cuda:0")


def test_kernel_outputs_are_named_and_placed() -> None:
    """
    A kernel argument the kernel writes is named ``out_*``, and every ``out_*`` argument is last.

    ``.claude/CLAUDE.md`` section 2.1's rule, checked from both sides after the first full sweep of
    ``kernels/`` found eight outputs wearing plain names (four literally ``out``) and three
    read-only inputs wearing the prefix (``claim_collapses`` read a *prior* kernel's outputs under
    their producer's names). In-place arguments and scratch / persistent-state buffers are exempt
    by section 3 and listed in ``_KERNEL_OUTPUT_ALLOWLIST``, which is staleness-checked.
    """
    _fail("kernel output-naming violation(s):", kernel_output_naming_problems())


def test_array_annotations_are_subscript_style() -> None:
    """
    An array annotation reads ``wp.array[T]``, never the pre-1.12 ``wp.array(dtype=T)``.

    ``.claude/CLAUDE.md`` section 1.2. Both spellings compile, so nothing but a check stops the old
    one from coming back with the next large module: it survived in ``algorithms/ball_pivoting.py``
    (98 of the 176), ``reconstruction.py`` and ``remesh.py`` long after the convention settled, and
    ``remesh.py`` carried both styles at once -- which is the state that leaves a reader unsure
    which one is current.
    """
    _fail("call-style array annotation(s):", array_annotation_style_problems())


def test_kernel_casts_use_the_warp_spelling() -> None:
    """
    A cast inside a kernel reads ``wp.int32(...)`` / ``wp.float32(...)``, never ``int`` / ``float``.

    ``.claude/CLAUDE.md`` section 1.3. The two spellings are the same Warp builtins and generate
    identical code, so nothing but a check keeps them from coexisting -- the tree carried 329
    ``int(wp.tid())`` against 640 ``wp.int32(...)``, and 46 sites in 41 kernels cast a thread index
    with one spelling and re-cast the same local with the other a few lines later.

    The asymmetry that makes it a rule: ``float(...)`` is a *hard compile error* inside a
    ``wp.Float``-generic function (``Input types must be the same, got ['float64', 'float32']``),
    so a bare one quietly forecloses ever genericising its function. Python-scope ``int`` /
    ``float`` are untouched -- ``wp.constant(wp.float32(float("nan")))`` at module scope is an
    ordinary Python call.
    """
    _fail("bare int()/float() cast(s) in kernel scope:", builtin_cast_problems())


def test_kernel_integer_division_uses_the_floor_spelling() -> None:
    """
    An integer division inside a kernel reads ``//``, never ``/``.

    ``.claude/CLAUDE.md`` section 1.5. On integers the two are the *same* operation in Warp -- both
    truncate toward zero, where CPython's ``//`` floors -- so this is legibility, not correctness:
    ``/`` on two ``int32``s reads as real division and a reader has to recover both operand types
    before they know the line truncates.

    A check rather than an edit because the defect recurred. The third pass converted eight sites
    and wrote the rule into ``CLAUDE.md``; ``algorithms/multigrid.py``, written afterwards,
    reintroduced two (``column = t / n_rows``, ``column = t / stride``), and the scan added for
    this test turned up four more in ``algorithms/blue_noise.py`` that a textual pass had missed.
    """
    _fail("integer division(s) spelled '/' in kernel scope:", integer_division_problems())


def test_integer_division_scan_ignores_float_operands() -> None:
    """
    Check 17 fires on declared integers only -- a float division is not a violation.

    The failure mode a conservative scan is protecting against is being switched off by the first
    person it annoys, so the negative cases are pinned here rather than left to the tree happening
    not to contain them: float / float, float / int, a ``wp.Scalar``-generic parameter the scan
    cannot type, and an element of a float array all stay silent, while the two-integer case in
    the same fixture is reported.
    """
    source = textwrap.dedent(
        """
        import warp as wp

        @wp.func
        def divisions(
            a: wp.float32,
            b: wp.int32,
            g: wp.Scalar,
            values: wp.array[wp.float32],
            counts: wp.array[wp.int32],
        ) -> wp.float32:
            i = wp.int32(wp.tid())
            float_by_float = a / a
            float_by_int = a / b
            generic_by_int = g / b
            element_by_int = values[i] / b
            int_by_int = counts[i] / b
            return float_by_float + float_by_int + generic_by_int + element_by_int + int_by_int
        """
    )
    tree = ast.parse(source)
    constants = _int_module_constants(tree)
    (function,) = _kernel_scope_functions(tree)
    scalars, arrays = _int_typed_names(function, constants)
    flagged = [
        ast.unparse(node)
        for node in ast.walk(function)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Div)
        and _is_int_expression(node.left, scalars, arrays)
        and _is_int_expression(node.right, scalars, arrays)
    ]
    assert flagged == ["counts[i] / b"]


def test_kernel_signatures_use_the_warp_types() -> None:
    """
    A kernel-scope argument or return is annotated ``wp.bool`` / ``wp.int32`` / ``wp.float32``.

    ``.claude/CLAUDE.md`` section 1.2. Warp resolves the bare names to the same types, so -- as with
    checks 16 and 17 -- nothing but a scan keeps the two spellings from coexisting: about a fifth
    of the tree's return annotations were the bare form when this was written, plus a dozen bare
    ``int`` / ``bool`` parameters.

    Only ``@wp.kernel`` / ``@wp.func`` signatures are read, which is what keeps it honest: a kernel
    *factory* is ordinary Python and its parameters are correctly plain (``reduce.blocks_1d``,
    ``neighbors._bvh_nearest_row_kernel``), and annotating those ``wp.int32`` would misdescribe
    where they run.
    """
    _fail("bare bool/int/float annotation(s) in kernel scope:", bare_annotation_problems())


def test_bare_annotation_scan_ignores_kernel_factories() -> None:
    """
    Check 18 reads kernel-scope signatures only -- a factory's plain ``int`` is not a violation.

    The negative case is pinned here rather than left to the tree happening not to contain one,
    because a factory that returns a kernel sits in the same file as the kernels it builds and is
    the obvious false positive. ``wp.Scalar`` / ``Any`` / ``wp.array[...]`` and a bare ``str`` stay
    silent too; ``str`` is left out of the mapping deliberately, since no kernel argument can be
    one, which makes it a further tell that the enclosing function is Python.
    """
    source = textwrap.dedent(
        """
        import warp as wp

        def blocks_1d(n: int) -> int:
            return (n + 255) // 256

        def row_kernel_factory(row_size: int, name: str):
            @wp.kernel(name=name)
            def row_kernel(values: wp.array[wp.float32], scale: float) -> None:
                values[wp.tid()] = values[wp.tid()] * scale

            return row_kernel

        @wp.func
        def is_short(length: wp.float32, limit: wp.float32) -> bool:
            return length < limit

        @wp.func
        def generic(value: wp.Scalar, other: Any, table: wp.array[wp.int32]) -> wp.bool:
            return value > table[0]
        """
    )
    tree = ast.parse(source)
    flagged = [
        f"{function.name}:{name}"
        for function in _kernel_scope_functions(tree)
        for name, annotation in _annotation_nodes(function)
        if isinstance(annotation, ast.Name) and annotation.id in {"bool", "int", "float"}
    ]
    assert flagged == ["is_short:->", "row_kernel:scale"]  # ast.walk is breadth-first


def test_kernel_scope_has_no_python_ternary() -> None:
    """
    A ``@wp.kernel`` / ``@wp.func`` body spells a conditional value ``wp.where(cond, a, b)``.

    ``.claude/CLAUDE.md`` section 1.5. A Python ternary (``a if cond else b``) compiles to the same
    code as ``wp.where``, so -- as with checks 16, 17 and 18 -- nothing but a scan keeps the two
    spellings apart. This is check 20, added because the sixth kernels pass reported this axis at
    zero and was wrong: two ternaries in ``kernels/intersection.py`` predate that pass by several
    review rounds and survived a scan built specifically to find them.
    """
    _fail("kernel-scope ternary(s):", kernel_scope_ternary_problems())


def test_generic_kernels_register_their_overloads() -> None:
    """
    A ``@wp.kernel`` generic over a dtype has its concrete overloads registered at import.

    ``.claude/CLAUDE.md`` section 2.5. Warp instantiates a generic kernel's overload lazily, on the
    first launch at each new dtype, and a module's hash covers the *instantiated* set -- so a
    lazily-created overload silently rebuilds every kernel in its module. Nothing fails when that
    happens, which is why it needs a check: the whole defect is a cost. Measured before the
    registrations went in, over one full-suite run: ``kernels.reduce`` rebuilt across 66 distinct
    module loads, ``laplacian`` 16, ``array`` 13, ``scatter`` 11, and 464 of the 1 269 directories
    in the Warp kernel cache were dead ``reduce`` hash links.

    This asserts only that a generic kernel has *some* overload registered, which is what catches
    the real-world defect -- a new generic kernel added with no ``_register_overloads`` entry. **It
    cannot tell whether the registered dtype set is complete**, and no cheap check can: proving that
    means launching the whole dispatch, and a missing dtype announces itself as a rebuild rather
    than a wrong answer. So a suddenly slow test is the symptom to read, per ``.claude/CLAUDE.md``
    section 15.1 -- the dtype belongs in the module's ``_register_overloads``.
    """
    unregistered = [
        key
        for module_name in _triwarp_kernel_modules()
        for key, kernel in wp.get_module(module_name).kernels.items()
        if kernel.is_generic and not kernel.overloads
    ]
    _fail("generic kernel(s) with no registered overload:", sorted(unregistered))


def _triwarp_kernel_modules() -> list[str]:
    """
    Import every ``triwarp.kernels`` sub-module and return the Warp module names they registered.

    Importing is the point, not a side effect: a kernel module Warp has never seen has no entry to
    inspect, and the sub-packages (``kernels/algorithms/``, ``kernels/heat/``) are only reached by
    the wrappers that use them, so a plain ``import triwarp`` leaves several unregistered.
    """
    import importlib
    import pkgutil

    import triwarp.kernels

    for info in pkgutil.walk_packages(triwarp.kernels.__path__, "triwarp.kernels."):
        importlib.import_module(info.name)
    return [name for name in list(_warp_user_modules()) if name.startswith("triwarp.kernels")]


def _warp_user_modules() -> Mapping[str, object]:
    """Warp's registry of user modules, which has no public accessor."""
    from warp._src.context import user_modules

    return user_modules


def test_public_functions_document_what_they_raise() -> None:
    """
    A public function with a ``raise`` in its own body documents a ``Raises`` block.

    ``.claude/CLAUDE.md`` section 4.3's "docstring, signature and body must agree", from the side
    where the body says more than the docstring. Delegated validation is not scanned -- 42 public
    functions correctly document a ``Raises`` their shared guard performs.
    """
    _fail("public function(s) raising without a Raises block:", undocumented_raise_problems())


@pytest.fixture
def example_namespace(
    icosahedron: tuple[tm.Trimesh, wp.Mesh], hemisphere: tuple[tm.Trimesh, wp.Mesh]
) -> dict[str, object]:
    """
    Bind every name the package's docstring examples use to a real object on the fixture's device.

    Every free name an example needs lives here. A new example that reaches for a name this
    namespace does not carry fails with a ``NameError``, which is the intended outcome: an example
    the suite cannot run is exactly the state that let two broken ones survive.
    """
    _, mesh_wp = icosahedron
    device = mesh_wp.device
    vertices, faces = mesh_wp.points, mesh_wp.indices
    # A second, *open* mesh: an example about boundaries has nothing to show on the closed fixture.
    _, open_mesh_wp = hemisphere
    queries = wp.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=wp.vec3, device=device
    )
    neighbor_idx, neighbor_distance = tw.neighbors.query_nearest(
        vertices, vertices, 4, backend="bvh"
    )
    # Thresholded against its own mean, not against zero: the fixture is translated to z + 2, so
    # ``> 0.0`` selects every face and leaves a region with no seam around it.
    centroids_np = tw.triangles.face_centroids(vertices, faces).numpy()
    face_mask = wp.array(
        centroids_np[:, 2] > centroids_np[:, 2].mean(), dtype=wp.bool, device=device
    )
    return {
        "tw": tw,
        "wp": wp,
        "np": np,
        "face_mask": face_mask,
        "warp_mesh": mesh_wp,
        "v": vertices,
        "f": faces,
        "open_v": open_mesh_wp.points,
        "open_f": open_mesh_wp.indices,
        "pts": queries,
        "origins": queries,
        "directions": wp.array([[1.0, 0.0, 0.0]] * queries.size, dtype=wp.vec3, device=device),
        "neighbor_idx": neighbor_idx,
        "neighbor_distance": neighbor_distance,
    }


@pytest.mark.parametrize(
    "example", [pytest.param(item, id=item.site) for item in docstring_examples()]
)
def test_docstring_examples_run(
    example: DocstringExample, example_namespace: dict[str, object]
) -> None:
    """
    Every fenced ``python`` block in a docstring executes against a real mesh.

    An example is the one piece of documentation that can be checked by running it, and both
    failures this test was written for were runtime ones invisible to ``ast.parse``:
    ``points.outlier_probability`` fed a NumPy bool array to ``array.flatnonzero``, and
    ``ray.contains_points`` compared a ``wp.array`` with ``0.0``, which Warp's Python-scope arrays
    do not support. Blocks containing a bare ``...`` are deliberate outlines and are skipped.
    """
    if example.is_sketch:
        pytest.skip(f"{example.site} is a deliberate sketch (bare ...), not runnable code")
    try:
        exec(compile(example.code, example.site, "exec"), dict(example_namespace))
    except Exception as error:  # noqa: BLE001
        pytest.fail(
            f"{example.site}: the documented example raises "
            f"{type(error).__name__}: {error}\n{example.code}",
            pytrace=False,
        )


def test_reference_comparisons_carry_a_class_label() -> None:
    """
    A test asserting against a reference library says which class the comparison is.

    ``.claude/CLAUDE.md`` section 7.4, and it is a gate rather than a convention because the
    convention has decayed twice: the lowercase ``class b`` spelling went from 21 occurrences to 0
    in one pass and back to 9 in the next, invisible to the grep section 7.4 prescribes because a
    human reads ``class B`` and ``Class B`` the same. It also closes a rarer and worse case --
    ``test_fill_min_weight_matches_meshlib`` carried a ``parity`` marker, compared a class-C
    statistic against MeshLib and had no docstring at all, which ruff cannot see because ``D103``
    is in the ignore list.

    It checks that a label is *present*, never that it is the right one. Choosing between A, B, C
    and D is a judgement about what transform the comparison needs, and section 14's rule holds
    here as everywhere: a scan can tell that a convention was not broken, not that a new name is a
    good one.
    """
    _fail("reference comparison(s) with no class label:", comparison_label_problems())


def test_comparison_label_scan_keys_on_asserts_not_on_fixture_unpacking() -> None:
    """
    Check 19 reads ``assert`` statements only -- unpacking a mesh fixture is not a comparison.

    Pinned because it is the difference between a gate that fires 0 times and one that fires 120:
    every mesh test in the suite writes ``mesh_tm, mesh_wp = icosphere``, so a scan keyed on the
    function body would demand a class label from every one of them and would be switched off by
    the first person it annoyed. The ``_np`` suffix is left out of
    ``api_conventions._REFERENCE_SUFFIXES`` for the same reason, measured at 290 hits.
    """
    source = textwrap.dedent(
        """
        def test_unpacks_a_fixture_only(icosphere) -> None:
            mesh_tm, mesh_wp = icosphere
            answer_wp = tw.measures.volume(mesh_wp.points, mesh_wp.indices)
            assert float(answer_wp) > 0.0

        def test_compares_against_the_reference(icosphere) -> None:
            mesh_tm, mesh_wp = icosphere
            volume_tm = mesh_tm.volume
            assert np.isclose(float(tw.measures.volume(mesh_wp.points, mesh_wp.indices)), volume_tm)
        """
    )
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)]
    assert _asserted_reference_names(functions[0]) == set()
    assert _asserted_reference_names(functions[1]) == {"volume_tm"}


# --- the lazy package surface ---------------------------------------------------------------
#
# ``triwarp/__init__.py`` resolves every submodule through a PEP 562 ``__getattr__`` so that
# ``import triwarp`` does not decorate 553 kernels. These four tests pin the contract that change
# rests on. The first is the one that matters: the cost regresses the moment any eager import is
# added back, and it regresses *silently*, because nothing else in the suite can see it -- by the
# time a test runs, the modules it needed are imported and the surface looks identical.


def test_importing_triwarp_pulls_in_no_kernel_modules() -> None:
    """
    ``import triwarp`` imports no kernel module, and so decorates no kernel.

    Not a library comparison: this is a property of triwarp's own import graph. Runs in a
    subprocess because the in-process answer is always "all of them" -- the test session has
    already imported what it needs.

    A single eager ``from triwarp.mesh import Trimesh`` in ``__init__.py`` is enough to fail this,
    which is exactly what it is for: that one line costs the better part of a second of
    ``import triwarp``, and Python imports a parent package before its child, so it costs the same
    for ``import triwarp.edges`` too.
    """
    probe = textwrap.dedent(
        """
        import sys
        import triwarp
        kernels = sorted(m for m in sys.modules if m.startswith("triwarp.kernels"))
        public = sorted(
            m for m in sys.modules if m.startswith("triwarp.") and ".kernels" not in m
        )
        print(len(kernels), len(public))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    n_kernels, n_public = (int(token) for token in completed.stdout.split())
    assert n_kernels == 0, f"import triwarp pulled in {n_kernels} kernel modules"
    assert n_public == 0, f"import triwarp pulled in {n_public} public modules"


def test_every_public_name_resolves_and_is_cached() -> None:
    """
    Every name in ``__all__`` resolves, and resolving it caches it into the package namespace.

    Not a library comparison. The caching half is what keeps ``tw.laplacian`` in a hot wrapper an
    ordinary global lookup rather than a ``__getattr__`` call, so it is part of the contract and
    not an implementation detail.
    """
    for name in tw.__all__:
        assert getattr(tw, name) is not None
        assert name in vars(tw), f"{name} resolved but was not cached into triwarp's namespace"
    assert tw.Trimesh.__name__ == "Trimesh"  # a class, not a submodule: the one special case


def test_unknown_attribute_raises_attribute_error() -> None:
    """
    An unknown name raises ``AttributeError``, not ``ImportError`` or ``ModuleNotFoundError``.

    Not a library comparison. This is what keeps ``hasattr`` and ``getattr(..., default)`` working
    against the package, which a bare ``importlib.import_module`` in ``__getattr__`` would break.
    """
    with pytest.raises(AttributeError):
        _ = tw.definitely_not_a_module
    assert not hasattr(tw, "definitely_not_a_module")


def test_dir_lists_the_whole_surface_before_it_is_touched() -> None:
    """
    ``dir(triwarp)`` lists every public name whether or not it has been resolved.

    Not a library comparison. Without the module ``__dir__``, a lazy package lists only what some
    earlier caller happened to touch, which is what makes one hard to explore interactively.
    Subprocessed for the same reason as the first test: in-process, everything is already resolved.

    ``__version__`` is expected alongside ``__all__`` rather than inside it: it resolves through
    the same lazy ``__getattr__`` and should be discoverable interactively, but it is not part of
    the star-import surface, so ``__all__`` is the wrong place for it.
    """
    probe = textwrap.dedent(
        """
        import triwarp
        print(" ".join(dir(triwarp)))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert sorted(completed.stdout.split()) == sorted([*tw.__all__, "__version__"])


def test_single_index_tid_carries_the_declarative_cast() -> None:
    """
    A single-index ``wp.tid()`` is assigned as ``wp.int32(wp.tid())``, the one cast the tree keeps.

    ``.claude/CLAUDE.md`` section 1.3. ``wp.tid()`` already returns ``wp.int32``, so both spellings
    generate identical code and the defect is invisible to the compiler and to the suite -- this is
    check 22, the fifth member of the family checks 16, 17, 18 and 20 belong to. It was written
    with the axis at 422 cast against 45 bare, and the drift was per *file* rather than scattered,
    which is the signature of a convention nothing was holding. A multi-index unpack cannot carry a
    cast and is out of scope by construction.
    """
    _fail("bare single-index wp.tid():", bare_tid_problems())


def test_bare_tid_scan_ignores_multi_index_unpacks() -> None:
    """
    Check 22 never flags ``i, j = wp.tid()``, which cannot carry a cast.

    Not a library comparison: this pins the negative case of the scan above, the way
    ``test_integer_division_scan_ignores_float_operands`` pins check 17's. 61 multi-index unpacks
    in ``kernels/`` are correct as they are, so a scan that keyed on the call rather than on the
    target shape would report the axis at 61 defects and get switched off.
    """
    source = textwrap.dedent(
        """
        import warp as wp

        @wp.kernel
        def two_index(out: wp.array2d[wp.int32]) -> None:
            i, j = wp.tid()
            out[i, j] = i

        @wp.kernel
        def one_index_bare(out: wp.array[wp.int32]) -> None:
            i = wp.tid()
            out[i] = i
        """
    )
    tree = ast.parse(source)
    flagged = [
        function.name
        for function in _kernel_scope_functions(tree)
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and _is_tid_call(node.value)
    ]
    assert flagged == ["one_index_bare"]


def test_no_uncitable_library_references_in_the_package() -> None:
    """
    Nothing under ``triwarp/`` names a library the shipped package may not cite.

    Not a library comparison: this is a licensing property of triwarp's own source. Two libraries
    are covered, for opposite reasons. MeshLib's licence restricts *use* rather than distribution
    of derivatives, and triwarp ships ``MIT OR Apache-2.0``, so an attribution comment here reads
    as a claim that a permissively licensed package is derived from a proprietary one; the rule has
    failed twice (89 references removed in one pass across 19 files, five back by the eighth
    kernels pass, two saying "port of" in the imperative). ``promesh`` is the mirror-image case: it
    carries **no licence file at all** and is not a published package, so a "port of" claim against
    it cites terms nobody here can read -- five such references had accumulated in ``holes.py``.
    Naming MeshLib in ``tests/`` and ``benchmarks/`` is correct and required (a comparison has to
    say what it compares against), so the scan stops at the package.
    """
    _fail("uncitable library reference(s) under triwarp/:", uncitable_reference_problems())


def test_uncitable_scan_catches_every_shape_it_has_seen_and_not_circumradius() -> None:
    r"""
    Check 21's pattern fires on every shape seen, and on nothing that merely reads like one.

    Not a library comparison: this pins both directions of a licensing scan, the way
    ``test_bare_tid_scan_ignores_multi_index_unpacks`` pins check 22's. Both halves cost a wrong
    answer once. The positives are the shapes actually found in the tree -- a C++ source file, a
    class, a function, the package, an import, and (for ``promesh``) the two comments that claimed
    a port. A pattern keyed on a handful of named symbols matches **none** of the file-name ones,
    which is why this one is wider. The negatives matter because the ``MR`` prefix is matched under
    ``IGNORECASE``: without its ``\b`` it hits *circu-mradius*, and without the scoped ``(?-i:)``
    it hits any word starting with "mr". They also pin the *replacement* wording for the promesh
    half, so a future pass cannot "fix" this scan by widening it back onto the algorithm's own
    vocabulary. A narrowed pattern is how this rule failed the first two times.
    """
    caught = [
        "# Fan weight (port of FanOptimizer::calcQueueElement_)",
        "# MRLaplacian.cpp",
        "# MRPointCloudTriangulationHelpers.cpp",
        "positionVertsSmoothlySharpBd: SPD umbrella system",
        "from meshlib import mrmeshpy as mm",
        # The promesh half: the package, and the one function a comment claimed to port.
        "(the Warp port of promesh's ``triangulate_boundaries``)",
        "Mirrors promesh's private helper of the same name.",
    ]
    ignored = [
        "circumradius over twice the inradius",  # the \b, under IGNORECASE
        "the mrunning total",  # the scoped (?-i:), under IGNORECASE
        "MeshLab's ``inradius/circumradius``",  # pymeshlab is GPL and may be named
        "Attene's lightweight repair pipeline",  # pymeshfix may be named too
        "the gap-bridging problem Barequet and Sharir (1995) pose",  # the replacement wording
        "a greedy minimal-perimeter correspondence over the two rims",
    ]
    for line in caught:
        assert _UNCITABLE_REFERENCES.search(line) is not None, line
    for line in ignored:
        assert _UNCITABLE_REFERENCES.search(line) is None, line


def test_repeatedly_mapped_kernel_funcs_declare_their_signatures() -> None:
    """
    A kernel module whose ``@wp.func`` is ``wp.map``'d from several sites declares its signatures.

    Not a library comparison: this is a property of triwarp's own module-hash chains. ``wp.map``
    names its generated module after the *unqualified* op and forks its hash per call signature, so
    an op reached at three signatures builds its module three times, each build containing every
    kernel accumulated so far. Over one suite run without the tables that is dozens of redundant
    cold builds; with them the load count sits at the floor of one per
    ``(module, device, block_dim)`` pair.

    This is check 23, and it is the same *kind* of check as
    ``test_generic_kernels_register_their_overloads``: it asserts a module which needs a table has
    one, never that the table is complete. Nothing cheap can prove completeness, and the cost of an
    incomplete table is a rebuild rather than a wrong answer -- the completeness gate is the load
    census, which is a clock measurement.
    """
    _fail(
        "kernel module(s) mapping an op from several sites with no declaration table:",
        map_declaration_problems(),
    )


def test_claude_references_name_a_section_that_exists() -> None:
    """
    A ``.claude/CLAUDE.md`` cross-reference resolves, and names a subsection where one exists.

    Not a library comparison: this is a property of triwarp's own prose. Check 24, and it is the
    staleness half of the gate rather than a convention half -- the references it catches were all
    *correct* when they were written and rotted when CLAUDE.md's Part I was renumbered.

    **What makes it a check rather than a grep is the second clause.** A broken ``N.M`` reference
    is caught by resolution alone, but the 111 sites this was written for were bare *chapter*
    numbers, every one of which still resolved: "section 4" was standing for section 3.5
    (``wp.map`` targets), 2.5 (overload registration), 2.7 (``wp.launch`` cannot pass a
    ``wp.Function``) and 3.7 (``triplet_buffers``' uninitialized tail) at once, and "section 6" for
    four different subsections of chapter 7. One number meaning several sections is invisible to a
    resolving check, so the rule is that a subsection is named wherever one exists.

    The exemption is structural rather than written: chapters 5, 6, 8, 9, 10 and 11 have no ``###``
    heading, so a bare number is their only citation and flagging it would flag the correct
    spelling. Reading that from the file's own headings is what keeps the allowlist empty -- an
    allowlist carrying six entries that are really one fact is how a check gets switched off.

    Two things it deliberately cannot see, both fixed by hand in the pass that added it: a
    reference that names the file in one sentence and the number in the next, and a bare
    "section N" that belongs to a *paper* rather than to CLAUDE.md (``kernels/remesh.py`` cites
    "Liepa 2003, section 3"). Widening the pattern to catch the first would misfire on the second.
    """
    _fail(
        "CLAUDE.md cross-reference(s) naming a stale or under-specified section:",
        claude_section_reference_problems(),
    )


def test_admonitions_stay_out_of_numpydoc_item_sections() -> None:
    """
    A ``!!!`` admonition sits in free prose, never between two entries of a section.

    Not a library comparison: this is a property of triwarp's own docstrings. Check 25, and like
    the section-number check above it is a staleness half rather than a convention half -- every
    one of the eight sites it was written for reads perfectly in the source and renders wrongly.

    **The failure is griffe's parse, and it was confirmed rather than assumed.** Loading a function
    whose ``Raises`` block holds an admonition returns *three* raises entries, the middle one
    carrying the literal string ``!!! note "Some caveat"`` where an exception annotation belongs --
    so the published page grows a row for a type that does not exist, with the admonition's body as
    its description, and the warning never renders as a warning. ``zensical build --strict`` stays
    clean throughout, because every cross-reference in the swallowed text still resolves.

    The eight were six ``Raises`` blocks, one ``Returns`` and one more ``Raises``, across
    ``holes``, ``proximity``, ``sample``, ``smoothing`` and ``voxels`` -- and the clustering is the
    reusable part: an author writes the caveat where the thought occurs, and the thought occurs
    while documenting what the function rejects. All eight moved to ``Notes`` with no loss of
    meaning, which is why the check ships with an empty allowlist and no allowlist machinery.
    """
    _fail("docstring section(s) holding a misplaced admonition:", admonition_placement_problems())


def test_warp_typed_constants_stay_out_of_host_arithmetic() -> None:
    """
    A ``wp.int32`` constant is a kernel argument, never an operand of Python-scope arithmetic.

    Not a library comparison: this is a property of triwarp's own source. Check 26, and it belongs
    to the same family as checks 16, 17, 18, 20 and 22 -- the spelling is legal, the answer is
    correct, and nothing but a scan sees it.

    **The cost is measured, not assumed.** ``warp._src.types.scalar_base.__add__`` is
    ``return warp.add(self, y)``, which is Warp's Python-scope builtin dispatch and runs
    ``inspect.signature().bind()`` per operand -- two to three orders of magnitude a Python float's
    operator. A ``wp.array`` slice taken with such bounds is worse than one dispatch, because
    ``__getitem__`` forms ``stop - start`` and ``strides * start`` itself, so a symmetric slice is
    three. That is what ``kernels/array.py``'s ``LOOP_CONDITION_VIEW`` / ``LOOP_PROGRESS_VIEW``
    exist to avoid.

    **The probe that shows it bites**, run against ``triwarp/graph.py`` and reverted: restoring
    ``state[LOOP_PROGRESS : LOOP_PROGRESS + 1]`` reports two problems (one per bound), the
    one-sided ``state[LOOP_PROGRESS :]`` reports one, and a bare ``LOOP_PROGRESS * 2`` reports one.
    The correct spellings stay silent -- ``int(LOOP_CONDITION) + 1`` is an ``ast.Call`` operand,
    and a constant forwarded to ``wp.launch(inputs=[...])`` is neither a ``BinOp`` nor a ``Slice``.

    Its two deliberate blind spots: a Warp *vector*'s arithmetic goes through ``_binary_op``, a
    Python component loop rather than builtin dispatch (a different and smaller hazard), and an
    explicit ``wp.length(...)`` / ``wp.cross(...)`` call at Python scope costs a dispatch too but
    is often the right spelling -- ``np.cross`` is measurably *slower* than ``wp.cross`` -- so
    flagging those would make most of the hits legitimate. The runtime census in CLAUDE.md
    section 15.11 is what covers the wider class; this check covers the part with no legitimate
    instance.
    """
    _fail("Warp-typed constant(s) in Python-scope arithmetic:", warp_host_arithmetic_problems())


def test_no_warp_sparse_triplet_builds_in_the_package() -> None:
    """
    No ``triwarp/`` module calls ``warp.sparse.bsr_from_triplets``.

    Not a library comparison: this is a property of triwarp's own source. Check 27. The probe that
    shows it bites: reverting ``laplacian.cotmatrix`` to its triplet build reports one problem.
    """
    _fail("warp.sparse triplet build(s) in the package:", triplet_build_problems())


def test_every_public_module_is_in_the_docs_nav() -> None:
    """
    ``docs/SUMMARY.md`` shelves every public module exactly once.

    Not a library comparison: this is a property of the docs source. Check 28. The probe that
    shows it bites: deleting the ``voxels`` line from the nav reports one problem.
    """
    _fail("docs nav out of sync with triwarp/:", docs_nav_problems())
