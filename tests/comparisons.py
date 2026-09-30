"""
Comparison helpers for parity tests, where the two sides agree only up to a known transform.

Kept separate from [`tests/conversions.py`](conversions.py) because these are not format
conversions: each one encodes a *reason* why two correct implementations disagree elementwise --
ordering, winding, sign, gauge, or the absence of any correspondence at all -- and that reasoning
should have one home rather than being re-derived per module.

The classes of comparison, and the bar each has to clear (see CLAUDE.md section 7.4):

- **A** direct ``np.allclose`` / ``np.array_equal``. No helper needed.
- **B** equal after a named transform, at full ``1e-5`` tolerance. Most helpers here are class B:
  the transform is exact, so loosening the tolerance to accommodate it means the transform was
  wrong.
- **C** a derived scalar or set distance, because no correspondence between the two answers exists.
  [`symmetric_surface_distance`][tests.comparisons.symmetric_surface_distance],
  [`symmetric_chamfer`][tests.comparisons.symmetric_chamfer],
  [`chamfer_two_sided`][tests.comparisons.chamfer_two_sided],
  [`hausdorff_two_sided`][tests.comparisons.hausdorff_two_sided] and
  [`fraction_within`][tests.comparisons.fraction_within] are the class-C machinery. A class-C assert
  must name the bug class it excludes and record its measured margin in the test docstring -- a
  threshold sitting at the measured value is a latent flake *and* a weak test.

!!! warning
    ``fraction_within(a, b, ...) > f`` is the easiest of these to make vacuous, because a fraction
    computed over marginal distributions is invariant to shuffling one side. Before trusting one,
    check that shuffling ``b`` makes it fail.
"""

from __future__ import annotations

import numpy as np
import trimesh as tm
import warp as wp
from scipy.spatial import cKDTree


def lexsort_rows(rows_np: np.ndarray) -> np.ndarray:
    """
    Sort rows into a canonical order so two unordered row sets can be compared elementwise.

    The workhorse transform for index tables -- edge lists, face lists, adjacency pairs -- where
    triwarp's parallel construction and a reference's serial one both produce the right set in
    different orders. Note this canonicalises the row *order*, not the entries within a row; sort
    those first (``np.sort(edges, axis=1)``) when the pair itself is undirected.

    An empty input is returned unchanged. ``np.lexsort`` raises ``TypeError: need sequence of keys
    with len > 0`` on anything with no *columns* -- a bare ``(0,)`` or a ``(3, 0)`` -- which is
    reachable from any test whose mesh has no edges or no boundary, and is why two of the five
    private copies this replaced carried the guard and three did not.
    """
    rows_np = np.asarray(rows_np)
    if rows_np.size == 0:
        return rows_np
    return rows_np[np.lexsort(rows_np.T[::-1])]


def assert_unordered_rows_equal(rows_a: np.ndarray, rows_b: np.ndarray) -> None:
    """Assert two row sets are equal as sets, ignoring row order."""
    sorted_a, sorted_b = lexsort_rows(rows_a), lexsort_rows(rows_b)
    assert sorted_a.shape == sorted_b.shape, (
        f"row counts differ: {sorted_a.shape} vs {sorted_b.shape}"
    )
    assert np.array_equal(sorted_a, sorted_b)


def assert_nonconstant(values: np.ndarray, tol: float) -> None:
    """
    Assert a numeric field actually varies, guarding a comparison against passing vacuously.

    The same shape of bug as an empty answer (section 7.4): a reference or a triwarp field that
    happens to be constant on its fixture makes a permuted result, an off-by-one gather, or a
    query/vertex index swap all pass. ``tol`` is the field's own spread threshold and has no
    universal default -- callers pass what CLAUDE.md's own comment at the original site measured.
    """
    spread = float(np.ptp(values))
    assert spread > tol, f"expected non-constant values, got ptp={spread:.3e} (tol={tol:.3e})"


def undirected_edges(faces_np: np.ndarray) -> np.ndarray:
    """
    Build the ``(n_faces * 3, 2)`` undirected edge list of a face array, each row min-first.

    The single most re-derived line in this suite -- eight sites across six files spelled it three
    different ways -- and the one that has to agree with itself, because every topological reference
    quantity below is a reduction of it. Rows repeat: an interior edge appears twice and a boundary
    edge once, which is the signal
    [`edge_multiplicity`][tests.comparisons.edge_multiplicity] reads. Deduplicate with
    ``np.unique(..., axis=0)`` when the *set* is what is wanted.

    Parameters
    ----------
    faces_np
        ``(n_faces, 3)`` vertex indices. A flat triwarp face buffer needs ``.reshape(-1, 3)`` first.
    """
    return np.sort(np.asarray(faces_np)[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)


def edge_multiplicity(faces_np: np.ndarray) -> np.ndarray:
    """
    Incident-face count per distinct undirected edge, as a flat array of counts.

    ``1`` is a boundary edge, ``2`` an interior one and ``>= 3`` a non-manifold one, so a single
    ``np.unique`` answers every edge-topology question a reference comparison asks. The counts are
    in ``np.unique``'s sorted-edge order, which is why this returns the counts alone -- anything
    needing the edges beside them should call ``np.unique`` on
    [`undirected_edges`][tests.comparisons.undirected_edges] directly.
    """
    return np.unique(undirected_edges(faces_np), axis=0, return_counts=True)[1]


def euler_characteristic(faces_np: np.ndarray) -> int:
    """
    ``V - E + F`` of a face array, counting only the vertices some face references.

    Ignoring unreferenced vertices is deliberate and is what makes this comparable with a
    reference's answer: an isolated vertex is not part of the surface whose topology is under test,
    and several libraries drop them silently on the way in (CLAUDE.md section 7.6 on igl's
    ``F.max() + 1`` family). Pass the mesh's own vertex count instead of using this if the
    unreferenced ones are the thing being measured.
    """
    faces_np = np.asarray(faces_np)
    n_vertices = len(np.unique(faces_np))
    n_edges = len(np.unique(undirected_edges(faces_np), axis=0))
    return n_vertices - n_edges + int(faces_np.shape[0])


def open_edge_count(faces_np: np.ndarray) -> int:
    """
    Count the undirected edges with exactly one incident face -- the boundary edges.

    Note this is *not* pyvista's ``n_open_edges``, which counts boundary **plus** non-manifold edges
    and reads 7 where this reads 6 on three faces sharing one edge (CLAUDE.md section 7.6).
    """
    return int((edge_multiplicity(faces_np) == 1).sum())


def boundary_loop_sizes(faces_np: np.ndarray, min_size: int = 3) -> list[int]:
    """
    Vertex counts of the boundary loops, traced in numpy, largest first.

    The host-side description of a mesh's holes that hole-filling assertions need: how many loops,
    and how many vertices each spans, so a fill can be checked to have added ``size - 2`` faces per
    loop. ``test_combine.py`` and ``test_holes.py`` each carried three private helpers that read the
    same numbers off [`boundary_loops`][triwarp.boundary.boundary_loops] instead -- byte-identical
    between the two files -- and the counts do not need a device round trip.

    Boundary edges are the multiplicity-1 rows of
    [`edge_multiplicity`][tests.comparisons.edge_multiplicity], and the loops are the connected
    cycles they form.

    Sizes come back sorted, so this is for *counts* and *multisets* of sizes only. A test pairing a
    loop's size with another per-loop quantity needs them in one consistent order and should read
    both off ``boundary_loops`` (see ``test_fill_fan_preserve_largest``, which indexes sizes by the
    argmax of the perimeters).

    Parameters
    ----------
    faces_np
        ``(n_faces, 3)`` vertex indices. A flat triwarp face buffer needs ``.reshape(-1, 3)`` first.
    min_size
        Drop loops shorter than this. The default matches what the hole fillers call *fillable*: a
        loop of one or two vertices spans no triangle.

    Raises
    ------
    ValueError
        If a boundary vertex has other than two incident boundary edges, which makes "the loop
        through it" ambiguous. Tracing on regardless would return a plausible wrong number, so this
        refuses rather than guessing.
    """
    edges_np = undirected_edges(faces_np)
    unique_np, counts_np = np.unique(edges_np, axis=0, return_counts=True)
    boundary_np = unique_np[counts_np == 1]
    if boundary_np.size == 0:
        return []

    neighbours: dict[int, list[int]] = {}
    for first, second in boundary_np.tolist():
        neighbours.setdefault(first, []).append(second)
        neighbours.setdefault(second, []).append(first)
    ambiguous = sorted(vertex for vertex, ends in neighbours.items() if len(ends) != 2)
    if ambiguous:
        raise ValueError(
            f"vertices {ambiguous[:8]} have other than two incident boundary edges, so their "
            "boundary loop is not well defined"
        )

    sizes = []
    unvisited = set(neighbours)
    while unvisited:
        start = min(unvisited)
        vertex, previous, size = start, -1, 0
        while True:
            unvisited.discard(vertex)
            size += 1
            first, second = neighbours[vertex]
            step = first if first != previous else second
            vertex, previous = step, vertex
            if vertex == start:
                break
        sizes.append(size)
    return sorted((size for size in sizes if size >= min_size), reverse=True)


def canonical_labels(labels_np: np.ndarray) -> np.ndarray:
    """
    Relabel a partition by first occurrence, so two labellings of it compare elementwise.

    Component ids are arbitrary names for a partition: triwarp's label-propagation returns a
    *representative element's* index per component, scipy returns ``0..k-1`` in discovery order and
    ``igl.facet_components`` returns ``0..k-1`` in its own. Renaming each label to the position of
    its first appearance is the transform that makes the three comparable without hiding a genuine
    disagreement about *which* elements share a component.

    Every current caller reaches this through
    [`same_partition`][tests.comparisons.same_partition], which is the boolean a component test
    actually asserts; this is the transform underneath it, kept public because a test that wants to
    *print* the packed labelling on a failure needs it directly.
    """
    labels_np = np.asarray(labels_np).ravel()
    _first, inverse = np.unique(labels_np, return_inverse=True)
    # np.unique's inverse is ordered by sorted label value, not by first appearance; rank the
    # first-appearance positions to get the discovery order.
    order = np.argsort(
        np.array([np.flatnonzero(inverse == i)[0] for i in range(inverse.max() + 1)])
    )
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return rank[inverse]


def same_partition(labels_a: np.ndarray, labels_b: np.ndarray) -> bool:
    """
    Whether two labellings induce the same partition, ignoring the label names.

    See [`canonical_labels`][tests.comparisons.canonical_labels] for why the names differ between
    every implementation of this quantity.
    """
    return bool(np.array_equal(canonical_labels(labels_a), canonical_labels(labels_b)))


def canonical_winding(faces_np: np.ndarray) -> np.ndarray:
    """
    Rotate each triangle to start at its smallest index, preserving orientation.

    Two triangulations with the same faces *and* the same winding compare equal after this; a face
    whose winding was flipped does not, because rotation cannot undo a reflection. That is what
    makes it the right canonicalisation for orientation-repair comparisons -- insensitive to the
    arbitrary choice of starting corner and sensitive to the thing under test.
    """
    faces_np = np.asarray(faces_np).reshape(-1, 3)
    roll = np.argmin(faces_np, axis=1)
    return np.take_along_axis(faces_np, (roll[:, None] + np.arange(3)) % 3, axis=1)


def assert_same_up_to_sign(
    vectors_a: np.ndarray, vectors_b: np.ndarray, atol: float = 1e-5
) -> None:
    """
    Assert two sets of directions agree up to a per-element sign flip, via ``|dot| == 1``.

    Eigenvector-valued answers -- fitted line and plane normals, principal curvature directions,
    estimated point normals -- have no canonical sign: the reference's solver may return ``-v``
    where triwarp returns ``v`` and both are correct. ``|dot| == 1`` is the gauge-invariant
    statement, and it stays a class-B assert at full tolerance because the transform is exact.

    Does **not** admit an arbitrary rotation. If the two sides disagree by more than a sign -- a
    tangent frame rotated about its normal -- the answer is gauge-dependent in a stronger sense and
    must be compared through a genuinely invariant quantity instead (see CLAUDE.md section 7.6 on
    potpourri3d's tangent spaces).
    """
    a = np.asarray(vectors_a, dtype=np.float64).reshape(-1, np.shape(vectors_a)[-1])
    b = np.asarray(vectors_b, dtype=np.float64).reshape(-1, np.shape(vectors_b)[-1])
    assert a.shape == b.shape, f"shapes differ: {a.shape} vs {b.shape}"
    dots = np.abs(np.einsum("ij,ij->i", a, b))
    assert np.allclose(dots, 1.0, atol=atol), (
        f"worst |dot| deviation {np.abs(dots - 1.0).max():.3e}"
    )


def assert_cyclic_permutation_equal(loop_a: np.ndarray, loop_b: np.ndarray) -> None:
    """
    Assert two closed loops list the same cycle, up to starting point and direction.

    A boundary loop is a cyclic sequence; where it starts and which way it runs are conventions, not
    results. Comparing after canonicalising both is the only way to test the part that matters (the
    adjacency order) without testing the part that does not.

    Every current caller reaches this through
    [`assert_same_loop_set`][tests.comparisons.assert_same_loop_set], which pairs the loops up first
    -- this is the single-loop form, kept public because a test comparing *one* named rim should not
    have to wrap it in a list. It is the inner half of that function, not a second way to do the
    same job.
    """
    a, b = np.asarray(loop_a).ravel(), np.asarray(loop_b).ravel()
    assert a.shape == b.shape, f"loop lengths differ: {a.size} vs {b.size}"
    if a.size == 0:
        return
    assert set(a.tolist()) == set(b.tolist()), "loops visit different vertices"
    start = int(np.flatnonzero(b == a[0])[0])
    forward = np.roll(b, -start)
    backward = np.roll(b[::-1], -int(np.flatnonzero(b[::-1] == a[0])[0]))
    assert np.array_equal(a, forward) or np.array_equal(a, backward), (
        "same vertices but a different cyclic order"
    )


def trimesh_outline_loops(mesh_tm: tm.Trimesh) -> list[np.ndarray]:
    """
    Decode ``Trimesh.outline()`` into one vertex-index array per closed boundary loop.

    Two conventions, and neither is visible from the returned object's type. The ``Path3D``
    entities index into ``Path3D.vertices``, which is the mesh's own vertex array unchanged -- so no
    remapping is needed, and this asserts that rather than assuming it. And a **closed entity
    repeats its first point as its last**, so the trailing duplicate is dropped; leaving it in makes
    every loop one longer than triwarp's and reads as an off-by-one in triwarp.

    Neither the loop order within the list nor the starting point and direction within a loop is
    defined by either library, so pair the results by lowest vertex index and compare with
    [`assert_cyclic_permutation_equal`][tests.comparisons.assert_cyclic_permutation_equal].

    Shared by the free-function comparison in ``tests/test_boundary.py`` and the ``Trimesh``
    container one in ``tests/test_mesh.py``, which time as separate benchmark groups.
    """
    outline_tm = mesh_tm.outline()
    assert np.allclose(outline_tm.vertices, mesh_tm.vertices)  # entities index the mesh's own array
    loops_tm = []
    for entity in outline_tm.entities:
        assert bool(entity.closed), "an open outline entity means the fixture is not a clean rim"
        points = np.asarray(entity.points)
        assert points[0] == points[-1]
        loops_tm.append(points[:-1])
    return loops_tm


def assert_same_loop_set(loops_a: list[np.ndarray], loops_b: list[np.ndarray]) -> None:
    """
    Assert two unordered collections of closed loops describe the same cycles.

    The list-level counterpart of
    [`assert_cyclic_permutation_equal`][tests.comparisons.assert_cyclic_permutation_equal]: pairs
    the loops by lowest vertex index (no library defines the order between loops -- triwarp ranks by
    length, trimesh by traversal) and then compares each pair up to starting point and direction.
    """
    assert len(loops_a) == len(loops_b), f"loop counts differ: {len(loops_a)} vs {len(loops_b)}"
    for loop_a, loop_b in zip(
        sorted((np.asarray(loop).ravel() for loop in loops_a), key=lambda loop: int(loop.min())),
        sorted((np.asarray(loop).ravel() for loop in loops_b), key=lambda loop: int(loop.min())),
        strict=True,
    ):
        assert_cyclic_permutation_equal(loop_a, loop_b)


def fraction_within(
    values_a: np.ndarray, values_b: np.ndarray, rtol: float = 5e-2, atol: float = 5e-2
) -> float:
    """
    Fraction of elements agreeing within a relative-plus-absolute band.

    For references that are elementwise comparable *in principle* but carry a few genuinely
    unreliable entries -- libigl's principal curvature near a degenerate ring, say -- where masking
    them out individually would encode the reference's bugs into the test.

    A threshold on this is class C and needs the shuffle probe: if
    ``fraction_within(a, rng.permuted(b))`` also clears the bar, the number is describing the
    marginal distributions rather than the correspondence, and the assert is vacuous.
    """
    a = np.asarray(values_a, dtype=np.float64).ravel()
    b = np.asarray(values_b, dtype=np.float64).ravel()
    assert a.shape == b.shape, f"shapes differ: {a.shape} vs {b.shape}"
    return float(np.mean(np.abs(a - b) <= atol + rtol * np.abs(b)))


def symmetric_chamfer(mesh_a: tm.Trimesh, mesh_b: tm.Trimesh, n_samples: int = 4000) -> float:
    """
    Mean symmetric surface distance between two meshes, via surface sampling.

    The class-C fallback when two reconstructions of the same shape have no vertex correspondence
    at all -- different algorithms, vertex counts, topology. It measures whether the two describe
    the same *surface*, which is the strongest statement available.

    !!! warning "There is a sampling noise floor; a threshold must clear it"
        The two point sets are drawn independently, so a mesh compared with *itself* does not score
        zero. Measured on ``icosphere(subdivisions=3)`` at the default ``n_samples``:
        ``symmetric_chamfer(m, m)`` is **0.028**, while the same mesh translated by 0.05 scores
        **0.040**. A threshold below the floor can never pass, and one just above it cannot
        distinguish a 0.05 displacement from none. Raise ``n_samples`` to lower the floor, and quote
        both the self-distance and the real distance in the test docstring.

    Blind to anything that preserves the surface: a winding flip, a vertex permutation, or a
    duplicated face all score zero. Pair it with a structural assert when those matter.

    See Also
    --------
    [`symmetric_surface_distance`][tests.comparisons.symmetric_surface_distance]
        The same claim measured against the other mesh's *surface* rather than against a second
        point sample, which removes the noise floor entirely. Prefer it for a new comparison;
        this one stays for the tests whose thresholds are calibrated against its floor.
    """
    rng = np.random.default_rng(0)
    sample_a, _ = tm.sample.sample_surface(mesh_a, n_samples, seed=int(rng.integers(1 << 30)))[:2]
    sample_b, _ = tm.sample.sample_surface(mesh_b, n_samples, seed=int(rng.integers(1 << 30)))[:2]
    a_to_b = cKDTree(sample_b).query(sample_a)[0].mean()
    b_to_a = cKDTree(sample_a).query(sample_b)[0].mean()
    return float(0.5 * (a_to_b + b_to_a))


def symmetric_surface_distance(
    mesh_a: tm.Trimesh, mesh_b: tm.Trimesh, n_samples: int = 2000
) -> tuple[float, float]:
    """
    Mean and worst-case symmetric distance from each mesh's surface to the other's, no noise floor.

    Same class-C claim as [`symmetric_chamfer`][tests.comparisons.symmetric_chamfer] -- "these two
    describe the same surface", for two answers with no vertex correspondence -- but each sample is
    measured against the other mesh's **surface** (``trimesh.proximity.closest_point``) instead of
    against a second random sample of it. That one change removes the floor: the chamfer's two point
    sets are drawn independently, so even for identical meshes the nearest *sample* sits a mean
    spacing away, whereas the nearest *surface* is at distance zero.

    !!! note "Why the chamfer's floor is what it is"
        For two independent uniform samples at density ``n / area`` the mean nearest-neighbour
        distance is ``0.5 * sqrt(area / n)``, which is what ``symmetric_chamfer(m, m)`` reports for
        a mesh against itself -- its own sample spacing. On a pair of Poisson reconstructions this
        function scores **0** for a mesh against itself against the chamfer's floor, so the real
        disagreement is a signal rather than a perturbation of one.

    Returns both statistics from one sampling pass because they fail differently and each is blind
    to the other's bug class: a **local** defect barely moves the mean -- a dent over 2 % of the
    vertices scores *below* the level at which two honest implementations differ -- while tripling
    the max; a **global** scale error moves both. Assert whichever the test's claim is about, and
    say which in the docstring.

    Expensive per call at the default ``n_samples`` -- an exact surface query, not a tree
    lookup -- so it is affordable per test but not inside a loop.

    Parameters
    ----------
    mesh_a, mesh_b
        Meshes to compare. Neither needs to be watertight: the query is unsigned.
    n_samples
        Area-uniform samples drawn per mesh. Fewer are needed than for the chamfer, there being no
        pairing noise to average out.

    Returns
    -------
    tuple[float, float]
        ``(mean, max)`` of the two-sided sample-to-surface distance.

    See Also
    --------
    [`symmetric_chamfer`][tests.comparisons.symmetric_chamfer]
        The sample-to-sample form, and its noise floor.
    [`hausdorff_surface_two_sided`][tests.comparisons.hausdorff_surface_two_sided]
        The worst-case-only form of this same claim, taking ``(vertices, faces)`` pairs.
    [`hausdorff_two_sided`][tests.comparisons.hausdorff_two_sided]
        Worst-case distance between two bare point sets, when there is no surface to query.
    """
    rng = np.random.default_rng(0)
    sample_a, _ = tm.sample.sample_surface(mesh_a, n_samples, seed=int(rng.integers(1 << 30)))[:2]
    sample_b, _ = tm.sample.sample_surface(mesh_b, n_samples, seed=int(rng.integers(1 << 30)))[:2]
    a_to_b = tm.proximity.closest_point(mesh_b, sample_a)[1]
    b_to_a = tm.proximity.closest_point(mesh_a, sample_b)[1]
    mean = 0.5 * (float(a_to_b.mean()) + float(b_to_a.mean()))
    return mean, float(max(a_to_b.max(), b_to_a.max()))


def chamfer_two_sided(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """
    Two-sided mean-squared Chamfer distance between two **point sets**, in pytorch3d's convention.

    The point-set counterpart of [`symmetric_chamfer`][tests.comparisons.symmetric_chamfer], which
    takes two *meshes* and samples them itself -- so it is what a comparison between two samplers
    that have already produced their clouds needs, and handing point arrays to the mesh form
    raises inside trimesh rather than doing something sensible.

    The sum of the two directions' mean **squared** nearest-neighbour distances, which is what
    ``pytorch3d.loss.chamfer_distance`` and [`triwarp.metrics.chamfer_points_to_points`]
    [triwarp.metrics.chamfer_points_to_points] both return, so a threshold calibrated here reads
    on the same scale as those.

    Prefer this over [`hausdorff_two_sided`][tests.comparisons.hausdorff_two_sided] when the claim
    is distributional. Measured on two independent 1 000-point samplings of ``icosphere(2)``: the
    mean statistic separates the same mesh from one scaled by 1.15 by a factor of **6.8** (0.00768
    against 0.05227), where the worst-case Hausdorff separates them by **1.4** (0.162 against
    0.228) -- one stray sample in a tail dominates the max and swamps the signal.
    """
    a = np.asarray(points_a, dtype=np.float64)
    b = np.asarray(points_b, dtype=np.float64)
    return float((cKDTree(b).query(a)[0] ** 2).mean() + (cKDTree(a).query(b)[0] ** 2).mean())


def hausdorff_two_sided(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """
    Two-sided Hausdorff distance between two point sets -- the worst-case counterpart to chamfer.

    Where [`symmetric_chamfer`][tests.comparisons.symmetric_chamfer] averages and so tolerates a few
    stray elements, this reports the single worst one. Use it when the claim is "no part of either
    answer is far from the other", e.g. comparing intersection curves or sliced boundaries.

    Goes through a ``cKDTree`` rather than a dense ``(n, m)`` distance matrix. That is the same
    answer at a fraction of the memory, and it is why the private copy it replaced in
    ``tests/test_intersection.py`` is gone: two curve samples of a few thousand points each is an
    eight-figure matrix for one scalar.

    See Also
    --------
    [`hausdorff_surface_two_sided`][tests.comparisons.hausdorff_surface_two_sided]
        The same worst-case statement between two *surfaces*, which needs no correspondence between
        the meshes and no shared vertex count.
    [`chamfer_two_sided`][tests.comparisons.chamfer_two_sided]
        The averaged counterpart over the same two point sets, and the better discriminator where
        the claim is distributional rather than worst-case.
    """
    a = np.asarray(points_a, dtype=np.float64)
    b = np.asarray(points_b, dtype=np.float64)
    return float(max(cKDTree(b).query(a)[0].max(), cKDTree(a).query(b)[0].max()))


def hausdorff_surface_two_sided(
    vertices_a: np.ndarray,
    faces_a: np.ndarray,
    vertices_b: np.ndarray,
    faces_b: np.ndarray,
    n_samples: int = 5000,
) -> float:
    """
    Worst-case two-sided distance between two mesh *surfaces*, via surface sampling.

    The claim is "no part of either surface is far from the other" -- the statement a remeshing or
    decimation test wants, where the output has a different vertex count, a different triangulation
    and no correspondence at all with its input, so only a set distance can be asserted.

    Distinct from [`hausdorff_two_sided`][tests.comparisons.hausdorff_two_sided] despite the name:
    that one is point-set to point-set and this one measures each sample against the other mesh's
    surface, so a coarse triangulation is not penalised for having few vertices. It is the
    worst-case half of
    [`symmetric_surface_distance`][tests.comparisons.symmetric_surface_distance], kept separate
    because it queries through ``trimesh.proximity.signed_distance`` and the thresholds in
    ``tests/test_remesh.py`` are calibrated against these exact numbers.

    !!! warning "``signed_distance`` wants a closed mesh"
        The sign comes from a containment test, so on an open surface the magnitude is still the
        distance but the query is doing more work than it needs to. Prefer
        [`symmetric_surface_distance`][tests.comparisons.symmetric_surface_distance] for a new
        comparison on open input.

    Parameters
    ----------
    vertices_a, faces_a, vertices_b, faces_b
        The two meshes, as ``(n, 3)`` positions and ``(n_faces, 3)`` indices. Arrays rather than
        ``tm.Trimesh`` because every caller holds a raw pair straight out of triwarp, igl, open3d or
        pymeshlab, and building a mesh at each call site would be noise.
    n_samples
        Area-uniform samples drawn per mesh, at fixed seeds so the result is reproducible.

    Returns
    -------
    float
        The larger of the two one-sided worst-case distances.
    """
    mesh_a = tm.Trimesh(vertices_a, faces_a, process=False)
    mesh_b = tm.Trimesh(vertices_b, faces_b, process=False)
    sample_a, _ = tm.sample.sample_surface(mesh_a, n_samples, seed=0)[:2]
    sample_b, _ = tm.sample.sample_surface(mesh_b, n_samples, seed=1)[:2]
    a_to_b = np.abs(tm.proximity.signed_distance(mesh_b, sample_a)).max()
    b_to_a = np.abs(tm.proximity.signed_distance(mesh_a, sample_b)).max()
    return float(max(a_to_b, b_to_a))


def bsr_arrays(matrix: object) -> list[np.ndarray]:
    """
    Return a BSR matrix as ``[offsets, columns, values]``, sliced to its *true* entry count.

    ``matrix.nnz`` is a stale capacity after a duplicate-emitting build -- ``cotmatrix`` sizes its
    storage for six keys per face plus one per vertex -- so everything past ``nnz_sync()`` is
    uninitialized memory and comparing it reports a difference that is not there.
    """
    n_entries = int(matrix.nnz_sync())
    return [
        matrix.offsets.numpy(),
        matrix.columns.numpy()[:n_entries],
        matrix.values.numpy()[:n_entries],
    ]


def comparable_arrays(value: object) -> list[np.ndarray]:
    """
    Return whatever a wrapper produced flattened into the arrays a comparison can walk.

    Handles the four return shapes the calls below produce -- an array, a BSR matrix, a nested
    tuple of either, and a scalar -- and yields nothing for a ``LinearOperator``, whose state is
    the matrix it wraps and is compared through that matrix instead.
    """
    if isinstance(value, wp.array):
        return [value.numpy()]
    if hasattr(value, "nnz_sync"):
        return bsr_arrays(value)
    if isinstance(value, tuple | list):
        return [array for item in value for array in comparable_arrays(item)]
    if isinstance(value, bool | int | float):
        return [np.asarray(float(value))]
    # A Warp vector or matrix *value* (`wp.vec3`, `wp.mat33d`): a ctypes array, so it matches
    # none of the branches above and would otherwise return `[]` -- which reads as "compared and
    # equal" at every call site that zips this against another list. `centroid` and `bounds` were
    # silently unchecked that way.
    if hasattr(value, "_wp_scalar_type_"):
        return [np.array(value, dtype=np.float64).ravel()]
    return []


# `vertex_face_adjacency` is documented as a CSR of *sets*, and its row order is nondeterministic
# (the scatter that fills it races), so it differs run to run on one mesh and an elementwise
# comparison reports a difference that is not there. Compare those rows with `csr_row_sets`.
SET_VALUED_CACHE_KEYS = frozenset({"vertex_face_adjacency"})


def csr_row_sets(csr: tuple[wp.array, wp.array]) -> list[frozenset[int]]:
    """Per-row index sets of a ``(values, offsets)`` CSR pair."""
    values, offsets = csr
    flat, bounds = values.numpy(), offsets.numpy()
    return [frozenset(flat[bounds[i] : bounds[i + 1]].tolist()) for i in range(len(bounds) - 1)]
