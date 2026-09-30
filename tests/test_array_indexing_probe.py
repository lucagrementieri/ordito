"""
Probe: what Python-scope gather on a ``wp.array`` does with a **non-contiguous index array**.

``.claude/CLAUDE.md`` section 3.4 turns on one platform fact, and it is the kind that cannot be
found by reading: Warp's Python-scope gather ``src[indices]`` reads the index buffer *as if it were
contiguous* and silently ignores a view's stride. A column of an ``(n, 2)`` edge table, a step
slice, a reversed view -- each of those is a legal ``wp.array`` that prints correctly on its own and
gathers the wrong elements, with no exception. And the corrupt read is **faster** than the correct
one, because it touches a contiguous prefix, so neither the suite nor a benchmark catches it.

Every gather in ``triwarp/`` therefore either passes a whole array, a contiguous prefix slice, or a
buffer explicitly ``wp.clone``d out of a view -- a discipline nothing enforced until this file
existed. Section 3.4 cited it as already present; it was not, and only stale ``__pycache__``
remnants of a one-off exploratory run survived.

These are capability probes, not regression tests for triwarp code: they pin what the *platform*
does, so a future conversion can check rather than guess, and so a Warp release that fixes any of
this is noticed rather than silently relied upon. See ``.claude/CLAUDE.md`` sections 3.4 and 10.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import triwarp as tw


def test_column_view_is_a_correct_array_on_its_own(device: str) -> None:
    """A strided column view reads back correctly -- which is what makes the hazard invisible."""
    edges_np = np.array([[0, 10], [1, 11], [2, 12], [3, 13], [4, 14]], dtype=np.int32)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    column_wp = edges_wp[:, 0]

    assert not column_wp.is_contiguous
    assert np.array_equal(column_wp.numpy(), edges_np[:, 0])


def test_gather_through_a_column_view_ignores_the_stride(device: str) -> None:
    """
    The finding: ``payload[edges[:, 0]]`` returns the flattened buffer's leading entries.

    No exception and no warning -- the gather reads ``[0, 10, 1, 11, 2]`` where the column says
    ``[0, 1, 2, 3, 4]``. The two asserts are written as "is the flat prefix" rather than "is not the
    column" so this test *inverts* into a fix notification: if a Warp release starts honouring the
    stride, this fails and section 3.4's rule can be revisited rather than silently kept.
    """
    payload_np = np.arange(20, dtype=np.float32)
    edges_np = np.array([[0, 10], [1, 11], [2, 12], [3, 13], [4, 14]], dtype=np.int32)
    payload_wp = wp.array(payload_np, dtype=wp.float32, device=device)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)

    gathered_wp = wp.empty(edges_np.shape[0], dtype=wp.float32, device=device)
    wp.copy(gathered_wp, payload_wp[edges_wp[:, 0]])

    flat_prefix_np = payload_np[edges_np.reshape(-1)[: edges_np.shape[0]]]
    assert np.array_equal(gathered_wp.numpy(), flat_prefix_np)
    assert not np.array_equal(gathered_wp.numpy(), payload_np[edges_np[:, 0]])


def test_cloning_the_index_view_first_is_correct(device: str) -> None:
    """The prescribed fix: densify the index buffer, then gather. This is what ``triwarp/`` does."""
    payload_np = np.arange(20, dtype=np.float32)
    edges_np = np.array([[0, 10], [1, 11], [2, 12], [3, 13], [4, 14]], dtype=np.int32)
    payload_wp = wp.array(payload_np, dtype=wp.float32, device=device)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)

    for column in (0, 1):
        dense_wp = wp.clone(edges_wp[:, column])
        assert dense_wp.is_contiguous
        gathered_wp = tw.array.gather(payload_wp, dense_wp)
        assert np.array_equal(gathered_wp.numpy(), payload_np[edges_np[:, column]])


def test_step_slice_index_also_ignores_the_stride(device: str) -> None:
    """A step slice is the same hazard as a column: contiguity, not rank, is what matters."""
    payload_np = np.arange(32, dtype=np.float32)
    indices_np = np.array([0, 2, 4, 6, 8, 10, 12, 14], dtype=np.int32)
    payload_wp = wp.array(payload_np, dtype=wp.float32, device=device)
    indices_wp = wp.array(indices_np, dtype=wp.int32, device=device)

    stepped_wp = indices_wp[::2]
    assert not stepped_wp.is_contiguous
    assert np.array_equal(stepped_wp.numpy(), indices_np[::2])

    gathered_wp = wp.empty(stepped_wp.size, dtype=wp.float32, device=device)
    wp.copy(gathered_wp, payload_wp[stepped_wp])
    assert np.array_equal(gathered_wp.numpy(), payload_np[indices_np[: stepped_wp.size]])
    assert not np.array_equal(gathered_wp.numpy(), payload_np[indices_np[::2]])


def test_contiguous_prefix_slice_index_is_safe(device: str) -> None:
    """The one view shape that *is* safe, and the reason the rule is not "never slice"."""
    payload_np = np.arange(32, dtype=np.float32)
    indices_np = np.array([7, 3, 11, 0, 29, 15], dtype=np.int32)
    payload_wp = wp.array(payload_np, dtype=wp.float32, device=device)
    indices_wp = wp.array(indices_np, dtype=wp.int32, device=device)

    prefix_wp = indices_wp[:4]
    assert prefix_wp.is_contiguous
    gathered_wp = tw.array.gather(payload_wp, prefix_wp)
    assert np.array_equal(gathered_wp.numpy(), payload_np[indices_np[:4]])


def test_indexed_assignment_is_unsupported(device: str) -> None:
    """
    The other half of section 3.4: scatter has no Python-scope form, so it stays a kernel.

    Asserted as a raise rather than left to a comment, so a Warp release that adds it is noticed.
    """
    values_wp = wp.zeros(8, dtype=wp.float32, device=device)
    indices_wp = wp.array(np.array([1, 3, 5], dtype=np.int32), dtype=wp.int32, device=device)
    # Deliberately a bare ``Exception``: what is being probed is *whether* it raises at all, and
    # pinning the type would make this fail on a Warp release that merely reworded the error.
    with np.testing.assert_raises(Exception):
        values_wp[indices_wp] = 1.0
