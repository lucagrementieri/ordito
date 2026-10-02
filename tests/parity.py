"""
Cross-suite parity bookkeeping: which benchmarked reference libraries are also *tested* against.

``benchmarks/`` times ordito against every reference in ``benchmarks/conftest.py``'s
``LIBRARIES`` -- ten CPU-only baselines plus ``pytorch3d``, which carries CUDA kernels of its own
and so takes two rows -- but only ever asserts shapes and
finiteness -- its ``pytest_generate_tests`` parametrizes over ``(mesh_name, library)``, so one
benchmark invocation holds exactly one library's result and *cannot* compare them. That leaves room
for a benchmark to race two implementations computing different things and report the ratio as if it
meant something.

This module reads both suites statically and pairs them up:

- a benchmark declares what it times with ``@pytest.mark.benchmark(group=...)`` +
  ``@pytest.mark.benchlibs(...)``, and declares a reference whose *result* is not comparable with
  ``@pytest.mark.noparity("<library>", reason="...")``;
- a correctness test declares what it proves with ``@pytest.mark.parity("<group>", "<library>")``,
  and declares a reference it tests but which is deliberately *not* timed with
  ``@pytest.mark.parity("<group>", "<library>", benchmarked=False, reason="...")``.

[`tests/test_parity.py`](test_parity.py) then fails the default test run on any benchmarked pair
that is neither covered nor exempt. Run ``python -m tests.parity`` for the full matrix.

Why a static ``ast`` scan rather than pytest collection
-------------------------------------------------------
Importing the benchmark modules would execute ``benchmarks/conftest.py`` -- a bare
``from meshes import ...``, a ``pytest_addoption`` that cannot be registered twice, and the
``open3d`` / ``pymeshlab`` / ``meshio`` imports. More decisively, a collection hook sees only
*selected* items, so ``pytest tests/test_edges.py`` or ``-k centroid`` would report almost every
pair as uncovered -- wrong in exactly the runs developers do most. A static scan is independent of
selection, ordering, import health and device, so the gate's verdict is identical however pytest was
invoked. This module therefore imports **nothing** outside the standard library.

What a green gate does *not* mean
---------------------------------
**The inputs differ.** Benchmarks run on scan meshes (``bunny``, ``dragon``, ``lucy``); tests run on
the ``tests/conftest.py`` fixtures. Parity on a 42-vertex icosahedron does not establish parity on
``bunny``, where degeneracies, unreferenced vertices and float32 thresholds bite.

**Equivalence may be partial.** The marker asserts that *a* test compares the two; the nature of the
comparison lives in the test body, where a reduction, a unit fix or a gauge projection may stand
between the two answers. Green proves a human looked, not that the semantics match.
"""

from __future__ import annotations

import ast
import functools
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent
_BENCHMARKS_DIR = _REPO_ROOT / "benchmarks"

# Markers this module reads. Anything spelled ``pytest.mark.<one of these>`` anywhere in a scanned
# file must be consumed by the structured pass below, or ``_unconsumed`` reports it as an error --
# that is how exotic placements (on a class, in a module-level ``pytestmark``, under an alias) are
# rejected without enumerating them.
_BENCHMARK_MARKERS = frozenset({"benchmark", "benchlibs", "noparity"})
_TEST_MARKERS = frozenset({"parity"})

# Keyword arguments accepted per marker, and the literal type each takes. Every value must be a
# plain literal of that type -- see ``_literal_args`` for why a computed one is an error.
_ALLOWED_KEYWORDS: dict[str, dict[str, type]] = {
    "benchmark": {"group": str},
    "benchlibs": {},
    "noparity": {"reason": str, "oracle": str},
    "parity": {"benchmarked": bool, "reason": str},
}

# A ``noparity`` reason has to carry its own weight: the justifying prose already exists in the
# benchmark module docstrings, so the path of least resistance is ``reason="see module docstring"``,
# which tells the next reader nothing. The bar is mechanical or it will not hold. A
# ``benchmarked=False`` parity claim clears the same bar, for the same reason.
_MIN_REASON_CHARS = 40
_MIN_REASON_WORDS = 6
_HOLLOW_REASON = re.compile(
    r"^(see\b|n/?a$|todo|tbd|different$|not comparable$|no parity$|timing only$)", re.IGNORECASE
)

# Reference-variable suffixes mandated by CLAUDE.md section 7.1, used by the anti-vacuity check.
# ``trimesh`` and ``scipy`` also allow ``_np``: several benchmark rows are hand-rolled NumPy
# stand-ins for cached trimesh properties, and every scipy oracle is plain NumPy in and out.
#
# ``moderngl`` is the one entry here that is **not** a benchmark library, and it is registered
# anyway on purpose. It cannot carry a timed row -- an OpenGL rasterization prices driver and
# context overhead rather than an algorithm, which ``benchmarks/test_texture.py`` records -- so its
# claims are all ``benchmarked=False``. But an *unregistered* name silently disables
# ``_library_trace`` below (it returns ``None`` when the library is absent from this table), so
# leaving it out would let a moderngl claim land on a test that never consults moderngl. Registering
# a test-only reference is how the anti-vacuity check keeps applying to it.
_LIBRARY_SUFFIXES: dict[str, frozenset[str]] = {
    "trimesh": frozenset({"_tm", "_np"}),
    "igl": frozenset({"_igl"}),
    "open3d": frozenset({"_o3d"}),
    "scipy": frozenset({"_np"}),
    "potpourri3d": frozenset({"_pp"}),
    "pymeshlab": frozenset({"_pml"}),
    "pyvista": frozenset({"_pv"}),
    "meshlib": frozenset({"_ml"}),
    "pymeshfix": frozenset({"_pmf"}),
    "moderngl": frozenset({"_gl"}),
    # ``pytorch3d`` is the only reference here with GPU kernels of its own, and it is registered
    # for those alone -- one ``LIBRARIES`` row (``pytorch3d-cuda``) and one entry here, since a
    # marker names the library *kind* rather than a row.
    "pytorch3d": frozenset({"_p3d"}),
    # ``numpy`` has carried timed pairs since ``test_reduce.py`` landed and was missing from this
    # table the whole time, which meant every one of its claims skipped the check below -- the same
    # hole the moderngl note above describes, but live rather than hypothetical. Verified: all 16
    # existing numpy claims pass on ``_np`` with no renames.
    "numpy": frozenset({"_np"}),
}

# The second half of the anti-vacuity signal: a test may consult a reference without ever naming
# the result, e.g. ``_assert_same_faces(v_wp, f_wp, tm.creation.cylinder(...))``. Calling into the
# library's own module is as good a trace as binding a suffixed variable, so either satisfies the
# check. Roots are matched exactly against the import aliases each suite actually uses.
_LIBRARY_ROOTS: dict[str, frozenset[str]] = {
    "trimesh": frozenset(
        {
            "tm",
            "tms",
            "tm_geometry",
            "tm_grouping",
            "tm_repair",
            "tm_proximity",
            "tm_reg",
            "tm_segments",
            "tm_traversal",
            "tm_intersections",
            "tm_curvature",
            "tm_points",
            "tm_sample",
        }
    ),
    "igl": frozenset({"igl", "igl_module"}),
    "open3d": frozenset(
        {"o3d", "trimesh_to_open3d", "points_to_open3d", "open3d_to_trimesh", "trimesh_to_open3d_t"}
    ),
    "scipy": frozenset(
        {"sp", "scipy", "spla", "KDTree", "cKDTree", "csgraph", "Delaunay", "ConvexHull"}
    ),
    "potpourri3d": frozenset({"pp3d"}),
    "pymeshlab": frozenset(
        {"ml", "trimesh_to_pymeshlab", "warp_to_pymeshlab", "points_to_pymeshlab"}
    ),
    "pyvista": frozenset({"pv", "trimesh_to_pyvista", "points_to_pyvista"}),
    "moderngl": frozenset({"moderngl", "gl_context"}),
    "numpy": frozenset({"np"}),
    "meshlib": frozenset(
        {
            "mm",
            "mn",
            "numpy_to_meshlib",
            "trimesh_to_meshlib",
            "warp_to_meshlib",
            "points_to_meshlib",
            "meshlib_to_trimesh",
        }
    ),
    "pymeshfix": frozenset(
        {
            "pymeshfix",
            "_meshfix",
            "numpy_to_pymeshfix",
            "trimesh_to_pymeshfix",
            "warp_to_pymeshfix",
            "pymeshfix_to_numpy",
            "pymeshfix_intersecting_faces",
        }
    ),
    "pytorch3d": frozenset(
        {
            "p3d_ops",
            "p3d_loss",
            "p3d_structures",
            "p3d_utils",
            "numpy_to_pytorch3d",
            "trimesh_to_pytorch3d",
            "warp_to_pytorch3d",
            "points_to_pytorch3d",
            "points_to_torch",
            "pytorch3d_to_numpy",
        }
    ),
}


@dataclass(frozen=True)
class Site:
    """Where a marker was found, as a clickable ``path:lineno`` relative to the repo root."""

    path: str
    lineno: int
    function: str

    def __str__(self) -> str:
        """Render as ``path:lineno`` so editors and terminals can jump to it."""
        return f"{self.path}:{self.lineno}"


@dataclass(frozen=True)
class Exemption:
    """A benchmarked reference whose result is declared not comparable with ordito's."""

    group: str
    library: str
    reason: str
    oracle: str | None
    site: Site


@dataclass(frozen=True)
class Claim:
    """A correctness test's declaration that it asserts ordito agrees with one reference."""

    group: str
    library: str
    site: Site
    names: frozenset[str]
    """Every name assigned in the test's body, for the anti-vacuity check."""

    roots: frozenset[str]
    """Every dotted-call root used in the body, for the anti-vacuity check."""

    benchmarked: bool = True
    """
    Whether the pair is expected to be timed in ``benchmarks/`` as well as tested here.

    ``False`` declares *"tested here, and deliberately not benchmarked"* -- the case where a
    reference is a legitimate correctness oracle at fixture size but too expensive to time at scan
    size. It exempts the claim from
    [`test_parity_markers_reference_known_pairs`](test_parity.py) and nothing else; the claim still
    counts as coverage, still has to name a reference variable, and still fails if the benchmark row
    comes *back*, which is what keeps the declaration from going stale in the other direction.
    """

    reason: str = ""
    """The written justification required alongside ``benchmarked=False``, and empty otherwise."""


@dataclass
class BenchmarkScan:
    """Everything the benchmark suite declares about what it times and what it cannot compare."""

    libraries: dict[str, frozenset[str]] = field(default_factory=dict)
    """Group name -> its ``benchlibs`` kinds, ``ordito`` included."""

    sites: dict[str, Site] = field(default_factory=dict)
    """Group name -> where its benchmark function is defined."""

    exemptions: list[Exemption] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def pairs(self) -> set[tuple[str, str]]:
        """
        Every ``(group, library)`` pair that a correctness test could meaningfully cover.

        A group whose ``benchlibs`` has no ``ordito`` entry contributes nothing: it prices two
        references against each other (``fast_marching_distance`` times potpourri3d against
        pymeshlab for an algorithm ordito deliberately does not implement), so "ordito agrees with
        this reference" is not a statement about it. Excluding those here rather than exempting them
        one by one keeps the rule general -- a group that loses its ordito branch by accident stops
        demanding coverage instead of silently keeping a stale exemption.
        """
        return {
            (group, library)
            for group, libraries in self.libraries.items()
            if "ordito" in libraries
            for library in libraries - {"ordito"}
        }


@dataclass
class TestScan:
    """Every ``parity`` claim the correctness suite makes."""

    claims: list[Claim] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def pairs(self) -> set[tuple[str, str]]:
        """The set of ``(group, library)`` pairs claimed by at least one test."""
        return {(claim.group, claim.library) for claim in self.claims}


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


def _marker_name(decorator: ast.expr, watched: frozenset[str]) -> str | None:
    """Return the watched marker name if ``decorator`` is exactly ``pytest.mark.<name>(...)``."""
    if not isinstance(decorator, ast.Call):
        return None
    parts = _dotted(decorator.func)
    if len(parts) == 3 and parts[0] == "pytest" and parts[1] == "mark" and parts[2] in watched:
        return parts[2]
    return None


def _literal_args(
    call: ast.Call, marker: str, where: str, errors: list[str]
) -> tuple[list[str], dict[str, str | bool]] | None:
    """
    Extract a marker's literal arguments, appending a message and failing on anything else.

    Positional arguments are always non-empty strings; each keyword takes the literal type
    ``_ALLOWED_KEYWORDS`` declares for it. Rejecting non-literals outright is what keeps a static
    scan honest: a computed marker argument would be invisible here, so it is an error rather than a
    blind spot. Implicitly concatenated string literals are folded into one ``ast.Constant`` by the
    parser and pass; f-strings (``ast.JoinedStr``) and ``.format()`` calls do not.
    """
    args: list[str] = []
    for arg in call.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value:
            args.append(arg.value)
        else:
            errors.append(
                f"{where}: pytest.mark.{marker} arguments must be non-empty string literals, "
                f"got {ast.unparse(arg)!r}"
            )
            return None

    allowed = _ALLOWED_KEYWORDS[marker]
    keywords: dict[str, str | bool] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            errors.append(f"{where}: pytest.mark.{marker} does not accept ** unpacking")
            return None
        if keyword.arg not in allowed:
            expected = ", ".join(sorted(allowed)) or "no keyword arguments"
            errors.append(
                f"{where}: pytest.mark.{marker} got unknown keyword {keyword.arg!r} "
                f"(accepts {expected})"
            )
            return None
        wanted = allowed[keyword.arg]
        literal = keyword.value.value if isinstance(keyword.value, ast.Constant) else None
        # ``isinstance(True, str)`` and ``isinstance("x", bool)`` are both False, so the second test
        # rejects a swapped literal in either direction without a per-type branch. The first is what
        # tells a type checker the value is one of the two the table can name.
        if not isinstance(literal, (str, bool)) or not isinstance(literal, wanted):
            errors.append(
                f"{where}: pytest.mark.{marker} {keyword.arg}= must be a {wanted.__name__} "
                f"literal, got {ast.unparse(keyword.value)!r}"
            )
            return None
        keywords[keyword.arg] = literal
    return args, keywords


def _unconsumed(
    tree: ast.Module, watched: frozenset[str], consumed: set[int], path: str
) -> list[str]:
    """
    Report every watched marker the structured pass did not consume.

    One adversarial walk in place of a list of special cases: a marker on a class, inside a
    module-level ``pytestmark``, on a nested function or a fixture, spelled through an alias
    (``@mark.parity``), or applied without parentheses all leave a ``pytest.mark.<name>`` attribute
    on a line the structured pass never visited.
    """
    errors: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or node.attr not in watched:
            continue
        parts = _dotted(node)
        if len(parts) != 3 or parts[0] != "pytest" or parts[1] != "mark":
            continue
        if node.lineno in consumed:
            continue
        errors.append(
            f"{path}:{node.lineno}: pytest.mark.{node.attr} must decorate a module-level "
            "'def test_*' directly, spelled '@pytest.mark."
            f"{node.attr}(...)'"
        )
    return errors


def _iter_modules(directory: Path) -> list[Path]:
    """Every ``test_*.py`` in ``directory``, sorted so scan output is deterministic."""
    return sorted(directory.glob("test_*.py"))


def _parse(path: Path, errors: list[str]) -> ast.Module | None:
    """Parse a module, recording a ``SyntaxError`` as a scan error rather than raising."""
    relative = path.relative_to(_REPO_ROOT).as_posix()
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as error:
        errors.append(f"{relative}:{error.lineno or 0}: could not parse ({error.msg})")
        return None


@functools.cache
def scan_benchmarks() -> BenchmarkScan:
    """
    Read every benchmark module's ``benchmark`` / ``benchlibs`` / ``noparity`` markers.

    A function carrying *neither* ``benchmark`` nor ``benchlibs`` is ignored, which is what leaves
    ``benchmarks/test_meshes.py`` (mesh-fixture invariants, no timing) out with no filename
    special-case. Carrying one but not the other is an error.
    """
    scan = BenchmarkScan()
    if not _BENCHMARKS_DIR.is_dir():
        return scan

    group_sites: dict[str, Site] = {}
    seen_exemptions: dict[tuple[str, str], Site] = {}

    for path in _iter_modules(_BENCHMARKS_DIR):
        relative = path.relative_to(_REPO_ROOT).as_posix()
        tree = _parse(path, scan.errors)
        if tree is None:
            continue

        consumed: set[int] = set()
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
                continue
            where = f"{relative}:{node.lineno}"
            group: str | None = None
            libraries: set[str] | None = None
            exemptions: list[tuple[str, str, str | None, int]] = []

            for decorator in node.decorator_list:
                marker = _marker_name(decorator, _BENCHMARK_MARKERS)
                if marker is None:
                    continue
                assert isinstance(decorator, ast.Call)
                consumed.add(decorator.func.lineno)
                extracted = _literal_args(decorator, marker, where, scan.errors)
                if extracted is None:
                    continue
                args, keywords = extracted

                if marker == "benchmark":
                    if args:
                        scan.errors.append(
                            f"{where}: pytest.mark.benchmark takes group= as a keyword, "
                            f"not positionally"
                        )
                    if "group" not in keywords:
                        scan.errors.append(f"{where}: pytest.mark.benchmark needs group=")
                    else:
                        group = str(keywords["group"])
                elif marker == "benchlibs":
                    if not args:
                        scan.errors.append(f"{where}: pytest.mark.benchlibs names no libraries")
                    libraries = set(args)
                else:  # noparity
                    if len(args) != 1:
                        scan.errors.append(
                            f"{where}: pytest.mark.noparity names exactly one library per marker "
                            f"(stack the decorator for several), got {len(args)}"
                        )
                        continue
                    oracle = keywords.get("oracle")
                    exemptions.append(
                        (
                            args[0],
                            str(keywords.get("reason", "")),
                            None if oracle is None else str(oracle),
                            node.lineno,
                        )
                    )

            if group is None and libraries is None and not exemptions:
                continue
            if group is None or libraries is None:
                scan.errors.append(
                    f"{where}: a benchmark needs both pytest.mark.benchmark(group=...) and "
                    "pytest.mark.benchlibs(...)"
                )
                continue

            site = Site(relative, node.lineno, node.name)
            if group in group_sites:
                scan.errors.append(
                    f"{where}: benchmark group {group!r} is already used at {group_sites[group]}; "
                    "group names are the cross-suite join key and must be unique"
                )
                continue
            group_sites[group] = site
            scan.libraries[group] = frozenset(libraries)
            scan.sites[group] = site

            for library, reason, oracle, lineno in exemptions:
                key = (group, library)
                if key in seen_exemptions:
                    scan.errors.append(
                        f"{relative}:{lineno}: duplicate pytest.mark.noparity for "
                        f"{group!r} / {library!r} (already at {seen_exemptions[key]})"
                    )
                    continue
                seen_exemptions[key] = site
                scan.exemptions.append(Exemption(group, library, reason, oracle, site))

        scan.errors.extend(_unconsumed(tree, _BENCHMARK_MARKERS, consumed, relative))

    # An exemption for a library the benchmark does not actually time is how one goes stale: someone
    # edits benchlibs and the noparity marker is left behind pointing at nothing.
    for exemption in scan.exemptions:
        timed = scan.libraries.get(exemption.group, frozenset())
        if exemption.library not in timed:
            scan.errors.append(
                f"{exemption.site}: pytest.mark.noparity({exemption.library!r}) but "
                f"{exemption.group!r} does not time it (benchlibs: "
                f"{', '.join(sorted(timed)) or 'none'})"
            )
    return scan


@functools.cache
def scan_tests() -> TestScan:
    """Read every ``pytest.mark.parity`` claim in the correctness suite."""
    scan = TestScan()
    for path in _iter_modules(_TESTS_DIR):
        relative = path.relative_to(_REPO_ROOT).as_posix()
        tree = _parse(path, scan.errors)
        if tree is None:
            continue

        consumed: set[int] = set()
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
                continue
            where = f"{relative}:{node.lineno}"
            names = frozenset(_assigned_names(node))
            roots = frozenset(_called_roots(node))
            groups_seen: set[str] = set()

            for decorator in node.decorator_list:
                marker = _marker_name(decorator, _TEST_MARKERS)
                if marker is None:
                    continue
                assert isinstance(decorator, ast.Call)
                consumed.add(decorator.func.lineno)
                extracted = _literal_args(decorator, marker, where, scan.errors)
                if extracted is None:
                    continue
                args, keywords = extracted
                benchmarked = bool(keywords.get("benchmarked", True))
                reason = str(keywords.get("reason", ""))
                if benchmarked and "benchmarked" in keywords:
                    scan.errors.append(
                        f"{where}: pytest.mark.parity benchmarked=True is the default and says "
                        "nothing; omit it, or pass benchmarked=False to declare the pair "
                        "deliberately untimed"
                    )
                    continue
                if benchmarked and reason:
                    scan.errors.append(
                        f"{where}: pytest.mark.parity reason= only applies alongside "
                        "benchmarked=False; a pair that is both tested and benchmarked needs no "
                        "justification"
                    )
                    continue
                if len(args) < 2:
                    scan.errors.append(
                        f"{where}: pytest.mark.parity needs a group and at least one library, "
                        f"got {args}"
                    )
                    continue
                group, libraries = args[0], args[1:]
                if group in groups_seen:
                    scan.errors.append(
                        f"{where}: pytest.mark.parity names group {group!r} twice; "
                        "list its libraries in one marker instead"
                    )
                    continue
                groups_seen.add(group)
                if len(set(libraries)) != len(libraries):
                    scan.errors.append(
                        f"{where}: pytest.mark.parity({group!r}) repeats a library: {libraries}"
                    )
                    continue
                site = Site(relative, node.lineno, node.name)
                for library in libraries:
                    scan.claims.append(
                        Claim(group, library, site, names, roots, benchmarked, reason)
                    )

        scan.errors.extend(_unconsumed(tree, _TEST_MARKERS, consumed, relative))
    return scan


def _called_roots(node: ast.FunctionDef) -> set[str]:
    """
    Collect the base name of every call and attribute chain in a test's body.

    ``tm.creation.cylinder(...)`` contributes ``tm``, ``mesh_tm.outline()`` contributes ``mesh_tm``.
    Half of the anti-vacuity signal: a test may consult a reference inline without ever binding its
    result to a name, which the suffix half alone would read as a ordito-only test.
    """
    roots: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, (ast.Call, ast.Attribute)):
            continue
        current: ast.expr = child
        while isinstance(current, (ast.Attribute, ast.Call, ast.Subscript)):
            current = current.func if isinstance(current, ast.Call) else current.value
        if isinstance(current, ast.Name):
            roots.add(current.id)
    return roots


def _assigned_names(node: ast.FunctionDef) -> set[str]:
    """
    Every name bound anywhere in a test's body, including tuple targets and comprehensions.

    Used only by the anti-vacuity check, which asks whether a ``parity``-marked test names a
    reference variable at all -- the suffix convention from CLAUDE.md section 7.1 is the one
    machine-readable trace that a second implementation was consulted.
    """
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            names.add(child.id)
        elif isinstance(child, ast.arg):
            names.add(child.arg)
    return names


def reason_problem(reason: str, library: str) -> str | None:
    """
    Describe what is wrong with a written ``reason=``, or ``None`` when it passes.

    Shared by both markers that carry one -- a benchmark's ``noparity`` exemption and a test's
    ``parity(..., benchmarked=False)`` declaration. They justify opposite omissions but the bar is
    the same, and having one implementation is what keeps it that way.
    """
    reason = " ".join(reason.split())
    if not reason:
        return "reason= is required"
    if len(reason) < _MIN_REASON_CHARS:
        return f"reason= is {len(reason)} characters; say why in at least {_MIN_REASON_CHARS}"
    if len(reason.split()) < _MIN_REASON_WORDS:
        return f"reason= is {len(reason.split())} words; at least {_MIN_REASON_WORDS} are needed"
    if _HOLLOW_REASON.match(reason):
        return f"reason= restates the situation instead of explaining it: {reason!r}"
    if reason.lower() == library.lower():
        return "reason= is just the library name"
    return None


def suffix_problem(claim: Claim) -> str | None:
    """
    Describe why a claim looks vacuous, or ``None`` when the test really consults the reference.

    A ``parity`` marker is a self-assertion, and the likeliest way for one to be wrong is to sit on
    a test that only compares ordito with itself -- a precomputed-argument shortcut, or an
    invariant check. Two signals count as consulting the reference, and either suffices:

    - a **name** carrying the CLAUDE.md reference-variable suffix (``_tm`` / ``_igl`` / ``_pp`` /
      ``_pml`` / ``_o3d``, plus ``_np`` where the oracle is hand-rolled NumPy). Matching on suffixes
      rather than on imports is deliberate: ``tests/test_convex.py`` compares against trimesh
      throughout without ever importing it, because the comparison arrives as ``mesh_tm`` from a
      fixture tuple;
    - a **call rooted at the library**, for the tests that pass the reference inline and never name
      it -- ``_assert_same_faces(vertices_wp, faces_wp, tm.creation.cylinder(...))`` throughout
      ``tests/test_creation.py``. Requiring a binding there would be asking for a variable that
      exists only to satisfy this check.
    """
    suffixes = _LIBRARY_SUFFIXES.get(claim.library)
    if suffixes is None:
        return None
    if any(name.endswith(suffix) for name in claim.names for suffix in suffixes):
        return None
    if claim.roots & _LIBRARY_ROOTS.get(claim.library, frozenset()):
        return None
    expected = " / ".join(f"*{suffix}" for suffix in sorted(suffixes))
    return (
        f"claims {claim.library!r} but neither binds a {expected} variable nor calls into it; "
        "a parity test must read the reference's answer, not compare ordito with itself"
    )


def uncovered_pairs() -> set[tuple[str, str]]:
    """Benchmarked pairs with neither a ``parity`` claim nor a ``noparity`` exemption."""
    benchmarks = scan_benchmarks()
    exempt = {(item.group, item.library) for item in benchmarks.exemptions}
    return benchmarks.pairs - scan_tests().pairs - exempt


def format_uncovered(pairs: set[tuple[str, str]]) -> str:
    """
    Render uncovered pairs as a worklist, grouped by the benchmark file that declares them.

    Summary first so it survives pytest's output clipping and makes progress measurable, then one
    block per benchmark module naming the sibling test module -- the two files a reader has to open
    to resolve the entry either way.
    """
    benchmarks = scan_benchmarks()
    by_library: dict[str, int] = defaultdict(int)
    by_module: dict[str, list[tuple[int, str, list[str]]]] = defaultdict(list)
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)

    for group, library in pairs:
        by_library[library] += 1
        site = benchmarks.sites[group]
        grouped[(site.path, group)].append(library)

    for (path, group), libraries in grouped.items():
        site = benchmarks.sites[group]
        by_module[path].append((site.lineno, group, sorted(libraries)))

    lines = [
        f"{len(pairs)} benchmarked (group, library) pairs have no correctness test "
        "and no exemption.",
        "  by library:  "
        + "   ".join(f"{name} {count}" for name, count in sorted(by_library.items())),
        "  by module:   "
        + "   ".join(
            f"{Path(path).stem} {len(entries)}"
            for path, entries in sorted(by_module.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:5]
        )
        + (f"   (+{max(0, len(by_module) - 5)} more)" if len(by_module) > 5 else ""),
        "",
        "Each pair is timed in benchmarks/ but nothing in tests/ asserts the two agree.",
        "Resolve each one of two ways:",
        '  cover  -> add @pytest.mark.parity("<group>", "<library>") to the test comparing them',
        '  exempt -> add @pytest.mark.noparity("<library>", reason="...") to the benchmark',
        "",
    ]
    for path, entries in sorted(by_module.items()):
        sibling = _TESTS_DIR / Path(path).name
        suffix = (
            f"(tests: {sibling.relative_to(_REPO_ROOT).as_posix()})" if sibling.exists() else ""
        )
        lines.append(f"{path:<44} {suffix}")
        for lineno, group, libraries in sorted(entries):
            lines.append(f"  :{lineno:<5} {group:<44} {', '.join(libraries)}")
    return "\n".join(lines)


def _main() -> None:
    """Print the full parity matrix: pairs, claims, exemptions and what is still missing."""
    benchmarks = scan_benchmarks()
    tests = scan_tests()
    covered = tests.pairs
    exempt = {(item.group, item.library): item for item in benchmarks.exemptions}
    pairs = benchmarks.pairs

    print(f"benchmark groups        {len(benchmarks.libraries)}")
    print(f"(group, library) pairs  {len(pairs)}")
    print(f"  covered               {len(pairs & covered)}")
    print(f"  exempt                {len(pairs & set(exempt))}")
    print(f"  uncovered             {len(uncovered_pairs())}")
    for label, errors in (("benchmarks", benchmarks.errors), ("tests", tests.errors)):
        if errors:
            print(f"\n{len(errors)} {label} scan error(s):")
            for error in errors:
                print(f"  {error}")

    print("\n--- matrix ---")
    for group, library in sorted(pairs):
        if (group, library) in covered:
            state = "COVERED"
        elif (group, library) in exempt:
            state = "EXEMPT "
        else:
            state = "MISSING"
        print(f"{state}  {group:<44} {library}")

    if exempt:
        print("\n--- exemptions ---")
        for (group, library), item in sorted(exempt.items()):
            oracle = f" [oracle: {item.oracle}]" if item.oracle else ""
            print(f"{group} / {library}{oracle}\n  {item.site}\n  {item.reason}")

    untimed = sorted(
        (claim for claim in tests.claims if not claim.benchmarked),
        key=lambda claim: (claim.group, claim.library),
    )
    if untimed:
        print("\n--- tested, deliberately not benchmarked ---")
        for claim in untimed:
            print(f"{claim.group} / {claim.library}\n  {claim.site}\n  {claim.reason}")

    missing = uncovered_pairs()
    if missing:
        print("\n--- uncovered ---")
        print(format_uncovered(missing))


if __name__ == "__main__":
    _main()
