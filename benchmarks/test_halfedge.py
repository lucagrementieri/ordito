"""
Benchmarks for ``ordito.halfedge``: edge twins and counter-clockwise vertex one-rings.

Two axes, and they measure different things:

* **scale** -- ``halfedge_twins`` is a hash, a radix sort and a pair-up pass, so its cost is the
  pure ``N`` one and this axis is the whole story for it.
* **valence** -- ``sphere_med`` against ``fan_hub``, with ``V`` and ``F`` pinned and the maximum
  vertex valence going 6 -> 40 960. ``vertex_one_rings`` gives one thread the whole rotation around
  one vertex, so the *widest* ring sets the launch's critical path however small the mesh is. This
  is the group to watch: a 40 960-halfedge serial walk in a single thread is the shape that made
  ``igl.principal_curvature`` take 110 s on this same mesh.

``halfedge_twins`` sits on the suite's host-side floor over most of the scale axis, so that axis is
reporting launch overhead until the largest mesh, while ``vertex_one_rings`` spreads by an order of
magnitude on the valence axis at pinned ``V`` and ``F``. That spread is inherent rather than a
defect: the rotation
around a vertex is a linked walk, so a valence-``k`` hub is ``k`` dependent steps that no amount of
parallelism removes. It is recorded here so a future change that makes it *worse* is visible.

``vertex_one_rings_scale`` re-times the same walk with twins precomputed over uniform valence-6
meshes, and is flat and launch-bound across a 64x face-count range. With the valence group that is
the whole cost model here -- ring width matters, mesh size does not.

Both are pure connectivity, so the vertex positions never enter and there is nothing to compare
against another library's *geometry*.

References
----------
None of the four reference libraries exposes an equivalent, so these are before/after
self-comparisons.

**trimesh** has ``Trimesh.face_adjacency`` and ``vertex_neighbors``, but neither is a halfedge
pairing: the first drops which corner of which face an adjacency came from (which is exactly the
information ``twins`` keeps), and ``vertex_neighbors`` is an unordered set per vertex, with no
rotational order -- the property that makes the one-ring useful for tangent spaces. Timing
``vertex_neighbors`` against ``vertex_one_rings`` would compare "group the neighbours" against
"order the neighbours", which is the harder half.

**libigl** has ``igl.triangle_triangle_adjacency``, whose ``(F, 3)`` ``TT``/``TTi`` pair carries the
same information as ``twins`` in a different layout. It is the one reference that could be lined up
here; it is left out because the closest ordito function is ``adjacency.face_adjacency``, where it
is now a row ([`test_adjacency.py`](test_adjacency.py)), and adding a second row for a reshaped copy
of the same computation would double-count it.

``igl.vertex_triangle_adjacency`` is **not** a ``vertex_one_rings`` reference either, for the reason
given for ``vertex_neighbors`` above: it returns the incident faces per vertex in a ``(VF, NI)`` CSR
with no rotational order, so timing it here would again compare "group the neighbours" against
"order them". It is the reference for the *unordered* CSR instead, in
[`test_adjacency.py`](test_adjacency.py).

**potpourri3d** builds geometry-central's halfedge mesh internally for every solver, and
``pp3d.edges`` is the only place it surfaces the result -- an undirected edge list, not the twin
map. That call is timed in [`test_edges.py`](test_edges.py). **open3d** and **scipy** have nothing
comparable.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import ordito as od
from conftest import BenchCase


@pytest.mark.benchmark(group="halfedge_twins")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito")
def test_halfedge_twins(bench_case: BenchCase) -> None:
    """Hash, radix sort and pair up: the ``N``-driven half of this module."""
    faces, n_vertices = bench_case.faces_wp, bench_case.n_vertices
    twins = bench_case.run(lambda: od.halfedge.halfedge_twins(faces, n_vertices=n_vertices))
    assert twins.shape == (faces.size,)


@pytest.mark.benchmark(group="require_matching_twins")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito")
def test_require_matching_twins(bench_case: BenchCase) -> None:
    """
    Validating a caller-supplied twin table: one check launch and the readback that raises on it.

    At ``halfedge_twins``'s points, so the two read side by side as what a caller supplying
    ``twins=`` pays to have it checked against building it. Ordito only: no reference exposes a
    caller-supplied twin table.
    """
    faces, n_vertices = bench_case.faces_wp, bench_case.n_vertices
    twins = od.halfedge.halfedge_twins(faces, n_vertices=n_vertices)
    bench_case.run(lambda: od.halfedge.require_matching_twins(faces, twins))
    assert twins.shape == (faces.size,)


@pytest.mark.benchmark(group="vertex_one_rings")
@pytest.mark.benchaxis("valence")
@pytest.mark.benchlibs("ordito")
def test_vertex_one_rings(bench_case: BenchCase) -> None:
    """One serial rotation per vertex: on the valence axis, where the widest ring dominates."""
    faces, n_vertices = bench_case.faces_wp, bench_case.n_vertices
    ring, offsets, _ = bench_case.run(
        lambda: od.halfedge.vertex_one_rings(faces, n_vertices=n_vertices)
    )
    assert offsets.shape == (n_vertices + 1,)
    assert ring.shape == (faces.size,)


@pytest.mark.benchmark(group="vertex_one_rings_scale")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito")
def test_vertex_one_rings_scale(bench_case: BenchCase) -> None:
    """The same walk over uniform valence-6 meshes, for the ``N`` slope without the hub."""
    faces, n_vertices = bench_case.faces_wp, bench_case.n_vertices
    twins = od.halfedge.halfedge_twins(faces, n_vertices=n_vertices)
    _, offsets, _ = bench_case.run(
        lambda: od.halfedge.vertex_one_rings(faces, twins=twins, n_vertices=n_vertices)
    )
    assert offsets.shape == (n_vertices + 1,)


@pytest.mark.benchmark(group="halfedge_mates")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito")
def test_halfedge_mates(bench_case: BenchCase) -> None:
    """
    The mate code of every halfedge, with the vertex count passed (bucketed on a large CUDA mesh).

    The pairing ``halfedge_twins`` is built on, at its points, so the two read side by side.
    Ordito only: the test oracle is a NumPy grouping of the halfedge rows.
    """
    faces, n_vertices = bench_case.faces_wp, bench_case.n_vertices
    mates = bench_case.run(lambda: od.halfedge.halfedge_mates(faces, n_vertices))
    assert mates.shape == (faces.size,)


@pytest.mark.benchmark(group="key_ordered_halfedges")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito")
def test_key_ordered_halfedges(bench_case: BenchCase) -> None:
    """
    One halfedge per undirected edge (origin below destination), read back in edge-key order.

    One class with no ``key_order`` given, so the row is a compaction plus the key sort of half the
    halfedges: the ``edges_unique`` row order reached from a flag scan. The scan is an input,
    built on the host outside the timed region. Ordito only: the test oracle is NumPy.
    """
    faces, n_vertices = bench_case.faces_wp, bench_case.n_vertices
    faces_np = bench_case.faces_np
    flags_np = (faces_np < faces_np[:, [1, 2, 0]]).reshape(-1)
    inclusive_np = np.cumsum(flags_np, dtype=np.int32)
    count = int(inclusive_np[-1])
    inclusive = wp.array(inclusive_np, dtype=wp.int32, device=bench_case.device)
    halfedges = bench_case.run(
        lambda: od.halfedge.key_ordered_halfedges(faces, inclusive, count, n_vertices=n_vertices)
    )
    assert halfedges.shape == (count,)
