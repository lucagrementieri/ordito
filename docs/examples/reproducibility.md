---
head_title: "Reproducible GPU sampling and sums: examples"
description: >-
  See ordito's deterministic setting at work on CUDA: a seeded surface sampler and floating-point sums that repeat bit for bit from call to call.
---

# Reproducible results

What the deterministic setting changes, seen on a sampler and on a sum. The
[Reproducible results](../reproducibility.md) page explains the setting.

## A seeded sampler, called twice {#rp1}

[`sample_surface`][ordito.sample.sample_surface] picks each sample's face by searching a CDF
of the face areas, which a `float32` scan builds. On CUDA that scan combines its blocks in
arrival order, so with the setting off two calls with the same seed can round a few CDF
entries differently, and a sample drawn next to one of those entries lands on the adjacent
face *in index order*, which on a scan can be across the model. Inside
[`deterministic`][ordito.config.deterministic] the scan takes a fixed order and the second call
returns the first call's points bit for bit. The red segments join a sample's two positions.

```python
import numpy as np

import ordito as od
from examples import data

vertices, faces = data.load("dragon", device)
pairs = {}
for enabled in (False, True):
    with od.config.deterministic(enabled):
        first, _ = od.sample.sample_surface(vertices, faces, 100_000, seed=7)
        second, _ = od.sample.sample_surface(vertices, faces, 100_000, seed=7)
    moved = np.any(first.numpy() != second.numpy(), axis=1)
    print(f"deterministic={enabled}: {moved.sum()} of 100000 samples moved")
    pairs[enabled] = (first.numpy()[moved], second.numpy()[moved])
```

```text title="Output"
deterministic=False: 329 of 100000 samples moved
deterministic=True: 0 of 100000 samples moved
```

![A seeded sampler, called twice](../assets/examples/rp1.webp)

The count of moved points changes from run to run with the setting off; it is zero on the CPU
device, which runs its blocks in turn, whatever the setting.

## Sums that repeat bit for bit {#rp2}

A floating-point sum depends on the order its terms are added in, and on CUDA the blocks of a
reduction commit in the order they finish. [`volume`][ordito.measures.volume] of the dragon,
called 64 times with the setting off, returns several values a few `float32` units in the last
place apart; with it on, one value. Per-group sums go further:
[`scatter_sum`][ordito.reduce.scatter_sum] sums each group exactly in fixed point and rounds
once, so with the setting on its result is also the same on the CPU and the CUDA device.

```python
import numpy as np
import warp as wp

import ordito as od
from examples import data

vertices, faces = data.load("dragon", device)
volumes = {}
for enabled in (False, True):
    with od.config.deterministic(enabled):
        volumes[enabled] = np.array([od.measures.volume(vertices, faces) for _ in range(64)])
    print(f"deterministic={enabled}: {np.unique(volumes[enabled]).size} distinct volumes")

rng = np.random.default_rng(0)
values = rng.normal(size=200_000).astype(np.float32)
groups = rng.integers(0, 16, size=values.size).astype(np.int32)
with od.config.deterministic(True):
    sums = {
        name: od.reduce.scatter_sum(
            wp.array(values, device=name), wp.array(groups, device=name), 16
        ).numpy()
        for name in ("cpu", device)
    }
print("CPU and CUDA group sums identical:", np.array_equal(sums["cpu"], sums[device]))
```

```text title="Output"
deterministic=False: 13 distinct volumes
deterministic=True: 1 distinct volumes
CPU and CUDA group sums identical: True
```

![Sums that repeat bit for bit](../assets/examples/rp2.webp)
