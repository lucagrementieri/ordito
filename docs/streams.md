---
head_title: "Using ordito beside PyTorch and other CUDA libraries"
description: >-
  How ordito orders its GPU work: Warp's current stream, PyTorch's default and side streams, graph recording, threads, freed memory and the two memory pools.
---

# Sharing the GPU with other libraries

ordito runs inside your process, beside whatever else uses the GPU: PyTorch, a renderer, your own
Warp kernels. The CUDA work of all of them is ordered by **streams**. This page states what ordito
assumes about streams and memory, and the one pattern that keeps it in order with PyTorch.

## The stream contract

ordito issues every launch, copy and allocation of a call on **Warp's current stream** for the
device: `wp.get_device().stream`, which `wp.ScopedStream` sets for a block.

- **Inputs** must be ready on that stream: anything that wrote them on another stream must be
  ordered before it.
- **Outputs** are ready on that stream when the call returns. A call that reads a result back
  to the host (a count, a convergence test) has waited for its own work; otherwise the work may
  still be queued, and the next thing on the same stream sees it finished.
- **Graphs** that ordito records for its loops replay on that stream.

Whoever sets the stream owns the ordering. ordito never picks a stream of its own for your work
and never synchronizes the device.

## With PyTorch

Make torch's current stream Warp's current stream for the block that calls ordito:

```python
import torch
import warp as wp

import ordito as od

if torch.cuda.is_available():
    vertices, faces = od.creation.icosphere(subdivisions=4, device="cuda:0")
    with wp.ScopedStream(wp.stream_from_torch()):
        smoothed = od.smoothing.filter_laplacian(
            vertices, faces, iterations=4, implicit_time_integration=True
        )
        smoothed_torch = wp.to_torch(smoothed)  # the same memory, no copy
    print(tuple(smoothed_torch.shape))  # (2562, 3)
```

Both libraries then queue on one stream, so each sees the other's earlier work finished without an
event or a synchronization, whether torch is on its default stream or inside a
`torch.cuda.stream(...)` block.

Torch's default stream is CUDA's *legacy* default stream, on which CUDA cannot record a graph.
ordito records its loops (solver iterations, remeshing rounds, sweeps) on a private stream of its
own whenever the current stream is a blocking one, the legacy stream included, and replays them on
yours, so every function works under it. Recording issues no work, so that private stream is
*non-blocking*: it never stands between the legacy stream and anything else. Calls into Warp's own
iterative solvers run on a private *blocking* stream instead, since their work runs there: it is
ordered against the legacy stream in both directions, exactly as if it had been issued there.

Without `wp.ScopedStream`, ordito runs on Warp's own stream, which is also blocking: its work is
still ordered against torch's default stream, but not against a torch side stream.

## Side streams and freed memory

A stream from `torch.cuda.Stream()` is *non-blocking*: nothing orders it against the legacy stream
or against Warp's own stream. Two consequences:

- Order the hand-off yourself when ordito and torch run on different streams
  (`wp.wait_stream`, `torch.cuda.current_stream().wait_stream(...)`), or run both on the side
  stream as above.
- Warp returns a freed array's memory on the legacy stream, which does not wait for a
  non-blocking one. Keep every array (and every tensor viewing one through `wp.to_torch`) alive
  until the work using it on a side stream has finished, or synchronize that stream before
  dropping it. Warp's own documentation covers this under *non-blocking streams*.

## Torch in another thread

While a graph is being recorded on a *blocking* stream, CUDA refuses work on the legacy default
stream from any thread: torch raises `cudaErrorStreamCaptureImplicit` ("operation would make the
legacy stream depend on a capturing blocking stream") and the recording fails with it. ordito's own
recordings never use a blocking stream, so torch can keep running on its default stream in another
thread while ordito runs in this one.

Two calls still record on a blocking stream: `linalg.solve_spd` when it does not batch, and
`smoothing.filter_laplacian`'s implicit solve when its fixed-point step does not apply, both of
which hand the loop to Warp's own solvers. If another thread runs torch on its default stream at
the same time as one of those, run ordito on a torch side stream instead, which is non-blocking:

```python
import torch
import warp as wp

import ordito as od

if torch.cuda.is_available():
    vertices, faces = od.creation.icosphere(subdivisions=4, device="cuda:0")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())  # the inputs were written there
    with torch.cuda.stream(side), wp.ScopedStream(wp.stream_from_torch(side)):
        smoothed = od.smoothing.filter_laplacian(
            vertices, faces, iterations=4, implicit_time_integration=True
        )
    torch.cuda.current_stream().wait_stream(side)  # before torch reads the result
    print(smoothed.shape)  # (2562,)
```

## Calling ordito from several threads

Do not call ordito from two threads at the same time. Warp keeps one current stream per device for
the whole process, not one per thread: `wp.ScopedStream` in one thread changes the stream every
other thread launches on, and a launch issued while another thread records a graph is recorded
into that graph instead of running. Two concurrent calls fail with CUDA errors, hang or crash the
process, whatever stream each thread set.

A threaded process shares one lock among everything that uses Warp, ordito included, and holds it
for the whole call, the `wp.ScopedStream` block around it included:

```python
import threading

import ordito as od

ORDITO_LOCK = threading.Lock()


def smooth(subdivisions: int, counts: list[int]) -> None:
    with ORDITO_LOCK:
        vertices, faces = od.creation.icosphere(subdivisions=subdivisions)
        smoothed = od.smoothing.filter_laplacian(vertices, faces, iterations=4)
        counts.append(smoothed.shape[0])


counts: list[int] = []
workers = [threading.Thread(target=smooth, args=(level, counts)) for level in (2, 3, 4)]
for worker in workers:
    worker.start()
for worker in workers:
    worker.join()
print(sorted(counts))  # [162, 642, 2562]
```

The lock costs no GPU throughput: the calls would queue on one stream anyway. Threads that do not
use Warp need no lock: PyTorch can run on its default stream in another thread throughout, as the
previous section describes.

## Graph capture by the caller

Do not call ordito inside your own `wp.ScopedCapture` or `torch.cuda.graph`. Most calls read a
size or a convergence flag back to the host, which a capture forbids, and the loops record graphs
of their own.

## Two memory pools

Warp allocates from the CUDA memory pool and PyTorch from its own caching allocator. Each keeps
memory it has freed for its own reuse, so a pipeline alternating between them can run out of
memory while both pools hold free blocks. ordito leaves both settings to you:

- `wp.set_mempool_release_threshold(device, threshold)` makes Warp return the pooled memory it
  holds beyond `threshold` (bytes, or a fraction of the device's memory) to the driver at the
  next synchronization, where torch can allocate it.
- `torch.cuda.empty_cache()` does the same for torch.
