"""
Tests for ``ordito._common``: graph recording under any current stream.

Not a library comparison: no reference library records Warp graphs, so each test is ordito
against ordito, the same call under Warp's own stream and under the legacy default stream,
byte for byte.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np
import pytest
import torch
import trimesh as tm
import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _common

_Call = Callable[[wp.array[wp.vec3], wp.array[wp.int32]], odt.ArrayNd]

# Calls that record graphs: ordito's own captures (the fixed-point smoothing passes, the heat
# solves; ``arap_parametrization`` is the call first reported failing on torch's default stream)
# and Warp's ``warp.optim.linear`` solvers, which record on the current stream themselves (the
# smoothing's BiCGSTAB fallback at ``lamb = 50``).
_CALLS: dict[str, _Call] = {
    "fixed_point": lambda vertices, faces: od.smoothing.filter_laplacian(
        vertices, faces, lamb=0.5, iterations=3, implicit_time_integration=True
    ),
    "bicgstab": lambda vertices, faces: od.smoothing.filter_laplacian(
        vertices, faces, lamb=50.0, iterations=3, implicit_time_integration=True
    ),
    "heat_geodesic": lambda vertices, faces: od.heat.heat_geodesic(
        vertices, faces, wp.array([0], dtype=wp.int32, device=vertices.device)
    ),
    "arap_parametrization": lambda vertices, faces: od.parametrization.arap_parametrization(
        vertices, faces, iterations=10
    ),
}


@pytest.mark.parametrize("call", sorted(_CALLS))
def test_recordings_run_on_the_legacy_stream(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh], call: str
) -> None:
    """
    Ordito against ordito: a call that records graphs answers alike on the legacy stream.

    The legacy default stream is the one ``wp.ScopedStream(wp.stream_from_torch())`` hands Warp
    while torch runs on its default stream, and CUDA cannot record on it: every one of these calls
    raised ``Warp error: unknown stream`` there until ``_common.recordable_stream`` moved the
    recording to a private stream. The answer must be the one the call gives on Warp's own stream.
    """
    _, mesh_wp = saddle_graded
    device = wp.get_device(mesh_wp.device)
    if not device.is_cuda:
        pytest.skip("only a CUDA device has streams")
    run = _CALLS[call]
    expected_np = run(mesh_wp.points, mesh_wp.indices).numpy()
    with wp.ScopedStream(wp.Stream(device, cuda_stream=0)):
        assert device.stream.cuda_stream == 0
        actual_np = run(mesh_wp.points, mesh_wp.indices).numpy()
    assert np.array_equal(actual_np, expected_np)


def test_a_recording_leaves_the_legacy_stream_to_other_threads(device: str) -> None:
    """
    Ordito against ordito: torch's default stream stays usable while another thread records.

    While a graph is recorded on a *blocking* stream, CUDA refuses work on the legacy default
    stream from every thread (``cudaErrorStreamCaptureImplicit``, "operation would make the legacy
    stream depend on a capturing blocking stream") and invalidates the recording. A process that
    runs torch in one thread and ordito in another (a model server's worker threads) hit it.
    ``_common.scoped_capture`` records on a private non-blocking stream instead, so the other
    thread's torch work runs and the recording replays what it recorded.
    """
    resolved = wp.get_device(device)
    if not resolved.is_cuda:
        pytest.skip("only a CUDA device has streams")
    values = wp.zeros(1024, dtype=wp.float32, device=device)
    recording = threading.Event()
    torch_done = threading.Event()
    failures: list[BaseException] = []

    def run_torch() -> None:
        # Legacy-stream work issued while the main thread's recording is open.
        assert recording.wait(timeout=30.0)
        try:
            ones = torch.ones(4096, device=device)
            assert float((ones + 1.0).sum().item()) == 2.0 * 4096
        except BaseException as error:  # noqa: BLE001 -- surfaced in the main thread
            failures.append(error)
        finally:
            torch_done.set()

    worker = threading.Thread(target=run_torch)
    worker.start()
    with (
        wp.ScopedStream(wp.Stream(resolved, cuda_stream=0)),
        _common.scoped_capture(resolved) as capture,
    ):
        values.fill_(2.0)
        recording.set()
        assert torch_done.wait(timeout=30.0)
    worker.join()
    assert not failures, failures
    graph = capture.graph
    assert graph is not None
    # Recorded, not run: the fill lands only when the graph replays.
    assert np.array_equal(values.numpy(), np.zeros(1024, dtype=np.float32))
    with wp.ScopedStream(wp.Stream(resolved, cuda_stream=0)):
        wp.capture_launch(graph)
        actual_np = values.numpy()
    assert np.array_equal(actual_np, np.full(1024, 2.0, dtype=np.float32))
