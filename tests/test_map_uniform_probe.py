"""
Probe: which scalar uniforms ``wp.map`` accepts alongside its array inputs.

``wp.map`` is documented as mixing scalars freely with arrays, but the interesting cases for this
package are the opaque handle types — a ``wp.uint64`` acceleration-structure id, which is what lets
a mesh or BVH query live inside a mapped ``@wp.func`` instead of a hand-written kernel. If that
does not work, the affected kernels have to stay kernels regardless of how elementwise their bodies
look.

**Finding.** It works, but the handle must be wrapped explicitly as ``wp.uint64(mesh.id)``.
``wp.Mesh.id`` is a plain Python ``int``, and ``wp.map`` infers ``int32`` for that, so passing it
bare fails with ``does not support the provided argument types vec3f, int32, float32`` — a type
error naming the coerced type rather than anything about handles, which is easy to misread as
"``wp.map`` cannot do mesh queries". ``wp.launch`` needs no such wrapping, so this only bites on
conversion.

These are capability probes, not regression tests for triwarp code: they exist so a future
conversion can check the platform rather than guess. See ``.claude/CLAUDE.md`` section 3.5.
"""

from __future__ import annotations

import numpy as np
import trimesh as tm
import warp as wp

import triwarp as tw
from tests.conversions import points_to_warp, warp_empty


@wp.func
def _scale_by(value: wp.vec3, factor: wp.float32) -> wp.vec3:
    return value * factor


@wp.func
def _snap_to_mesh(point: wp.vec3, mesh_id: wp.uint64, max_dist: wp.float32) -> wp.vec3:
    query = wp.mesh_query_point_no_sign(mesh_id, point, max_dist)  # pyright: ignore[reportArgumentType]  # vec3 is vec3f
    if query.result:  # pyright: ignore[reportAttributeAccessIssue]  # the stub's MeshQueryPoint has no fields
        # Warp's own stub inconsistency: the builtin is declared to return ``vec3f`` while
        # ``wp.vec3`` re-exports a separate declaration of the same runtime class
        # (``wp.vec3 is wp.vec3f``). Kernel DSL is why ``triwarp/kernels`` is excluded outright.
        return wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)  # pyright: ignore[reportReturnType, reportAttributeAccessIssue]
    return point


@wp.func
def _classify(count: wp.int32) -> wp.int32:
    if count == 0:
        return wp.int32(0)
    return wp.int32(1)


def test_map_accepts_a_float_scalar_uniform(device: str) -> None:
    """The documented baseline: a plain float scalar mixed with an array input."""
    points_wp = points_to_warp(np.arange(9, dtype=np.float32).reshape(3, 3), device)
    out_wp = warp_empty(3, wp.vec3, device)
    wp.map(_scale_by, points_wp, wp.float32(2.0), out=out_wp)
    assert np.allclose(out_wp.numpy(), np.arange(9, dtype=np.float32).reshape(3, 3) * 2.0)


def test_map_accepts_a_uint64_mesh_id_and_queries_inside_the_func(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    ``wp.map`` carries a ``wp.Mesh.id`` as a scalar uniform, and mesh queries work in the func.

    Note the explicit ``wp.uint64(...)``: without it the id arrives as ``int32`` and the call
    fails. That is the whole reason this probe exists.

    This is what makes ``remesh._reproject_pass``-shaped work expressible as a map: the whole body
    is one closest-point query per vertex against a fixed mesh.
    """
    _, mesh_wp = icosahedron
    device = mesh_wp.device
    # Push every vertex outward, then snap back: the result must return to the surface.
    original_np = mesh_wp.points.numpy()
    pushed_wp = points_to_warp(original_np * 1.1, device)
    snapped_wp = warp_empty(pushed_wp.size, wp.vec3, device)

    wp.map(_snap_to_mesh, pushed_wp, wp.uint64(mesh_wp.id), wp.float32(1.0), out=snapped_wp)

    # Snapped points sit on the mesh, so their distance to it is ~0 while the pushed ones are not.
    _, distance_wp, _ = tw.proximity.closest_point_on_mesh(
        mesh_wp.points, mesh_wp.indices, snapped_wp
    )
    assert float(np.max(np.abs(distance_wp.numpy()))) < 1e-4
    assert not np.allclose(snapped_wp.numpy(), pushed_wp.numpy())


def test_map_writes_every_element_even_where_a_kernel_would_have_skipped(device: str) -> None:
    """
    A mapped func must return a value for every element, unlike a kernel that writes conditionally.

    Kernels that only wrote on one branch relied on a pre-zeroed destination. Converting one means
    the func has to produce the "untouched" value explicitly — after which the destination can be
    ``wp.empty`` rather than ``wp.zeros``.
    """
    counts_wp = wp.array(np.array([0, 3, 0, 7], dtype=np.int32), dtype=wp.int32, device=device)
    out_wp = warp_empty(4, wp.int32, device)  # deliberately not zeroed
    wp.map(_classify, counts_wp, out=out_wp)
    assert np.array_equal(out_wp.numpy(), np.array([0, 1, 0, 1], dtype=np.int32))
