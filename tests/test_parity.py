"""
The parity gate: a reference library that is *benchmarked* must also be *tested* against.

``benchmarks/`` asserts only shapes and finiteness, so nothing there establishes that ordito and
the reference it is timed against compute the same thing. These tests close that loop by pairing the
benchmark suite's ``benchmark`` / ``benchlibs`` markers with the correctness suite's ``parity``
markers, and failing when a benchmarked pair is neither covered by a test nor explicitly exempted
with a written justification. See [`tests/parity.py`](parity.py) for the scanner and for the limits
of what a green gate proves.

One test per failure mode rather than one big one, so the failing test's *name* says what kind of
mistake was made. Deliberately not parametrized over the pairs: that would add hundreds of
always-green items to every run, and a pair that *disappeared* would silently lose its check instead
of failing.
"""

from __future__ import annotations

import difflib

import pytest

from tests.parity import (
    format_uncovered,
    reason_problem,
    scan_benchmarks,
    scan_tests,
    suffix_problem,
    uncovered_pairs,
)

# Static scans of the source tree: the CPU pass of ``python -m tests.devices`` deselects them.
pytestmark = pytest.mark.device_agnostic


def _fail(headline: str, problems: list[str]) -> None:
    """Fail with a counted headline and one indented line per problem, without a traceback."""
    if not problems:
        return
    body = "\n".join(f"  {problem}" for problem in sorted(problems))
    # pytrace=False: the traceback runs through the scanner and says nothing useful, and pytest
    # truncates long assertion explanations in non-verbose runs -- which is exactly when the full
    # list is wanted.
    pytest.fail(f"{len(problems)} {headline}\n{body}", pytrace=False)


def test_benchmark_markers_are_wellformed() -> None:
    _fail("problem(s) with benchmark markers:", scan_benchmarks().errors)


def test_parity_markers_are_wellformed() -> None:
    _fail("problem(s) with parity markers:", scan_tests().errors)


def test_benchmark_suite_is_discoverable() -> None:
    """
    Guard the scan itself: a silently empty benchmark scan would make the gate vacuously green.

    Neither ``tests/`` nor ``benchmarks/`` ships in the wheel (``packages.find`` includes only
    ``ordito*``), so a missing directory is a legitimate packaged-tree situation and skips. A
    directory that exists but yields nothing is a broken scanner or a bad path, and fails.
    """
    from tests.parity import _BENCHMARKS_DIR  # pyright: ignore[reportPrivateUsage]

    if not _BENCHMARKS_DIR.is_dir():
        pytest.skip("benchmarks/ is not present in this tree")
    assert scan_benchmarks().libraries, "benchmarks/ has modules but the scan found no groups"


def test_parity_markers_reference_known_pairs() -> None:
    """
    Catch typos in the join key, which is the dominant failure mode of string-keyed bookkeeping.

    A ``parity`` marker naming a group that does not exist, or a library that group does not
    benchmark, would otherwise sit in the tree looking like coverage while contributing none.

    Claims declaring ``benchmarked=False`` are exempt here and checked by the two tests below
    instead: they name a pair that is *known* not to be timed, and saying so is the point.
    """
    benchmarks = scan_benchmarks()
    problems: list[str] = []
    for claim in scan_tests().claims:
        if not claim.benchmarked:
            continue
        timed = benchmarks.libraries.get(claim.group)
        if timed is None:
            suggestions = difflib.get_close_matches(claim.group, benchmarks.libraries, n=3)
            hint = (
                f"; did you mean {', '.join(repr(s) for s in suggestions)}?" if suggestions else ""
            )
            problems.append(f"{claim.site}: no benchmark group named {claim.group!r}{hint}")
        elif claim.library not in timed:
            problems.append(
                f"{claim.site}: group {claim.group!r} does not benchmark {claim.library!r} "
                f"(benchlibs: {', '.join(sorted(timed - {'ordito'})) or 'none'})"
            )
    _fail("parity marker(s) name a pair that is not benchmarked:", problems)


def test_exemptions_are_not_contradicted() -> None:
    """
    An exemption and a real test for the same pair means one of them is stale.

    In practice it is the exemption: somebody wrote the comparison and left the ``noparity`` behind,
    so the benchmark now claims a non-comparability its own test suite disproves.
    """
    covered = scan_tests().pairs
    problems = [
        f"{item.site}: noparity({item.library!r}) on {item.group!r}, but a parity test already "
        "covers that pair -- drop the exemption"
        for item in scan_benchmarks().exemptions
        if (item.group, item.library) in covered
    ]
    _fail("exemption(s) contradicted by an existing parity test:", problems)


def test_exemption_oracles_are_covered() -> None:
    """
    An ``oracle=`` redirect is a claim the gate checks, not a note.

    Several references are exempt because they wrap an implementation another library already
    provides -- pymeshlab's harmonic and LSCM filters wrap libigl, its screened Poisson is the
    same Kazhdan code open3d wraps. Naming the library that *is* the oracle turns "not independent
    evidence" into something verifiable: that oracle's own pair must be covered, or the exemption
    has redirected to nothing.
    """
    benchmarks = scan_benchmarks()
    covered = scan_tests().pairs
    problems: list[str] = []
    for item in benchmarks.exemptions:
        if item.oracle is None:
            continue
        if item.oracle == item.library:
            problems.append(f"{item.site}: oracle= names {item.library!r}, the exempted library")
        elif item.oracle not in benchmarks.libraries.get(item.group, frozenset()):
            problems.append(
                f"{item.site}: oracle={item.oracle!r} is not benchmarked by {item.group!r}"
            )
        elif (item.group, item.oracle) not in covered:
            problems.append(
                f"{item.site}: oracle={item.oracle!r} but ({item.group!r}, {item.oracle!r}) "
                "has no parity test either -- the redirect points at nothing"
            )
    _fail("exemption oracle redirect(s) unresolved:", problems)


def test_exemption_reasons_are_substantive() -> None:
    """
    Hold exemptions to a written justification, mechanically.

    The prose already exists in the benchmark module docstrings, so the path of least resistance
    is ``reason="see module docstring"`` -- which records that somebody noticed without telling the
    next reader anything. A length-and-content floor is crude, but it is the only kind of bar that
    holds without a reviewer in the loop.
    """
    problems = [
        f"{item.site}: noparity({item.library!r}) on {item.group!r}: {problem}"
        for item in scan_benchmarks().exemptions
        if (problem := reason_problem(item.reason, item.library)) is not None
    ]
    _fail("exemption reason(s) too thin:", problems)


def test_untimed_parity_reasons_are_substantive() -> None:
    """
    Hold a ``benchmarked=False`` declaration to the same written bar as a ``noparity`` exemption.

    The two markers justify opposite omissions -- "timed but not comparable" against "compared but
    not timed" -- and both are a human's assertion that the gap is deliberate. A declaration that
    does not say *why* the pair is untimed is indistinguishable from one added to silence a red
    gate, which is the failure this whole module exists to prevent.
    """
    problems = [
        f"{claim.site}: parity({claim.group!r}, {claim.library!r}, benchmarked=False): {problem}"
        for claim in scan_tests().claims
        if not claim.benchmarked
        and (problem := reason_problem(claim.reason, claim.library)) is not None
    ]
    _fail("untimed parity declaration(s) too thin:", problems)


def test_untimed_parity_claims_are_really_untimed() -> None:
    """
    The staleness guard in the other direction: a declared-untimed pair that *is* timed.

    ``benchmarked=False`` suppresses a real check, so it has to expire on its own. If somebody
    restores the benchmark row -- open3d's screened Poisson becomes affordable, a ``benchlibs`` edit
    adds the library back -- the declaration is now a false statement quietly exempting a pair that
    no longer needs it, and only this test notices.
    """
    benchmarks = scan_benchmarks()
    problems = [
        f"{claim.site}: parity({claim.group!r}, {claim.library!r}, benchmarked=False), but "
        f"{claim.group!r} does time {claim.library!r} now -- drop the declaration"
        for claim in scan_tests().claims
        if not claim.benchmarked
        and claim.library in benchmarks.libraries.get(claim.group, frozenset())
    ]
    _fail("untimed parity declaration(s) contradicted by a benchmark:", problems)


def test_parity_claims_read_a_reference_variable() -> None:
    """
    Anti-vacuity: a ``parity`` marker must sit on a test that actually reads the reference's answer.

    A marker is a self-assertion, and the likeliest way for one to be wrong is to land on a test
    that only compares ordito with itself -- a precomputed-argument shortcut, say. The
    reference-variable suffixes CLAUDE.md section 7.1 already mandates (``_tm`` / ``_igl`` /
    ``_pp`` / ``_pml`` / ``_o3d``, plus ``_np`` where the oracle is hand-rolled NumPy) are the one
    machine-readable trace that a second implementation was consulted.

    This is the most heuristic check here, and the most likely to need adjusting rather than
    satisfying: if it starts firing on a legitimate test, widen the suffix table in
    ``tests/parity.py`` rather than renaming a variable to appease it.
    """
    problems = [
        f"{claim.site} ({claim.site.function}): {problem}"
        for claim in scan_tests().claims
        if (problem := suffix_problem(claim)) is not None
    ]
    _fail("parity claim(s) that never bind a reference variable:", problems)


def test_every_benchmarked_pair_has_parity_coverage() -> None:
    """
    The gate proper: every benchmarked reference is either tested against or exempted.

    A failure here is a worklist, not a defect -- each entry is one benchmark comparison whose two
    sides have never been shown to compute the same thing.
    """
    missing = uncovered_pairs()
    if missing:
        pytest.fail(format_uncovered(missing), pytrace=False)
