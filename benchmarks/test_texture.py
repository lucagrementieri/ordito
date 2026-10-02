"""
Benchmarks for ``ordito.texture``.

Two inverse pairs, with opposite cost drivers:

* **Rasterize** (``rasterize_attribute``, ``rasterize_discrete_attribute``) — scatter: one thread
  per triangle walks the pixels its UV-space bounding box covers and interpolates barycentrically.
  Cost is *total covered area in pixels*, so it scales with ``resolution ** 2`` far more strongly
  than with the face count, and it is sensitive to UV-space triangle size — many small triangles
  amortize the per-triangle setup badly. The discrete variant does strictly more per pixel: it
  accumulates per-label weights and takes an argmax instead of one lerp.
* **Remap** (``remap_attribute_from_uv``, ``remap_discrete_attribute_from_uv``) — gather: one thread
  per *vertex*, one bilinear (``order=1``) or nearest (``order=0``) fetch. Cost is the vertex count,
  essentially independent of resolution beyond cache behaviour, so these are two orders of magnitude
  cheaper and are launch-latency bound on the smaller meshes. Timing both orders isolates the
  interpolation from the gather.

Axis: the **scan sweep** for the face-count half, plus a **resolution** sweep of 512 against 2048 on
every group. Those are the module's two independent sizes and they pull in opposite directions, so
pinning either hides half the story: the rasterizers should show a ~16x step across the resolution
pair and the remappers none at all beyond cache effects. A rasterizer that fails to scale
quadratically, or a remapper that does, is the signal this pair exists to produce.

UV coordinates
--------------
The scan meshes carry no UV map, and computing a real one per case is not viable: an LSCM or
harmonic parametrization needs disk topology (these are near-closed) and would time the solver
rather than the rasterizer.

So UVs are the **vertices' xy coordinates normalized to the unit square** — a projection, not an
injective parametrization, so on a closed mesh front and back land on the same pixels and the
documented "lowest face index wins" rule resolves the overlap. That does not distort what is
measured: the rasterizer's cost is the number of (triangle, covered pixel) pairs, which a projection
produces just as faithfully, and the overlap additionally exercises the contention path a real atlas
would not. It *would* matter for a correctness comparison, which is why the parity tests use
synthetic injective UVs instead.

References
----------
**No CPU baseline is registered, and the reason is scope rather than absence.** Neither trimesh,
libigl nor open3d has a UV-space attribute rasterizer or sampler.

**pymeshlab has both directions** — ``transfer_attributes_to_texture_per_vertex`` forward,
``compute_color_from_texture_per_vertex`` inverse — so all five groups have a counterpart. What
keeps them from being rows, and from being parity oracles, was probed rather than assumed, and the
blocker is narrower than "the output is an image": **``pymeshlab.Image`` is write-only from
Python**, its whole public surface being ``width``, ``height`` and ``save``, so neither direction
has an in-memory route.

* **Forward.** The bake can only leave Python as a file, so a comparison goes through an 8-bit PNG.
  For a ``float32`` raster that makes the quantization a *difference* between the two sides rather
  than something both share — a class-D shape rather than a tolerance.
* **Inverse.** The sampler needs a texture *attached to the mesh*, and there is no array route to
  that either, only loading a mesh file that references an image file. That path does work, and
  reveals a second convention: **MeshLab samples with wrap addressing.** On a 32-column ramp
  texture, vertices at ``u = 1.0`` come back as 0 where the texel is 248 — ``u = 1`` wraps to column
  0 — while an interior sample reads correctly. On a *random* texture that makes all four UV corners
  return the same value, which reads as a broken oracle rather than an addressing rule.

So the inverse pair is reachable at 8 bits and on strictly interior UVs, at the cost of writing an
OBJ, an MTL and a PNG per case; the forward pair is not reachable at array level at all. Its
parameter surface is also large — the bake takes a texture size, a sampling density and an
interpolation mode that do not map one-to-one onto ``resolution`` and ``order`` — but that is the
smaller obstacle.

The references the *correctness* tests use are not benchmarkable baselines either: the forward
rasterizers are checked against **moderngl (OpenGL)**, which would time driver and context overhead
rather than an algorithm, and the inverse samplers against ``scipy.ndimage.map_coordinates``, which
samples a plain grid with no notion of UV or mesh topology — the right correctness oracle for the
interpolation and the wrong baseline for the operation. So these are before/after self-comparisons.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import ordito as od
import ordito.typing as odt
from conftest import BenchCase, skip_larger_than

# Texture sizes: 4x the pixels between the two points, so the rasterizers' quadratic term and the
# remappers' independence from resolution both read directly off the pair.
_RESOLUTIONS = [512, 2048]

# Resolution of the cached source textures the remap groups sample from.
_RESOLUTION = 1024

# Number of distinct labels for the discrete pair.
_N_CLASSES = 8

_uv_cache: dict[tuple[str, str], wp.array[wp.vec2]] = {}
_attribute_cache: dict[tuple[str, str], odt.Array2dFloat32] = {}
_labels_cache: dict[tuple[str, str], wp.array[wp.int32]] = {}
_image_cache: dict[tuple[str, str], odt.Array3dFloat32] = {}
_class_image_cache: dict[tuple[str, str], odt.Array2dInt32] = {}


def _uv_wp(bench_case: BenchCase) -> wp.array[wp.vec2]:
    """Vertex xy normalized to the unit square — a projection, not an atlas (see docstring)."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _uv_cache:
        xy = bench_case.vertices_np[:, :2]
        span = np.where(xy.max(axis=0) - xy.min(axis=0) > 0.0, xy.max(axis=0) - xy.min(axis=0), 1.0)
        uv = (xy - xy.min(axis=0)) / span
        _uv_cache[key] = wp.array(
            np.ascontiguousarray(uv, dtype=np.float32), dtype=wp.vec2, device=bench_case.device
        )
    return _uv_cache[key]


def _attribute_wp(bench_case: BenchCase) -> odt.Array2dFloat32:
    """``(n_vertices, 3)`` float attribute (the positions themselves) — the usual bake payload."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _attribute_cache:
        _attribute_cache[key] = wp.array(
            np.ascontiguousarray(bench_case.vertices_np, dtype=np.float32),
            dtype=wp.float32,
            device=bench_case.device,
        )
    return _attribute_cache[key]


def _labels_wp(bench_case: BenchCase) -> wp.array[wp.int32]:
    """``(n_vertices,)`` int labels cycling through ``_N_CLASSES``."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _labels_cache:
        labels = np.arange(bench_case.n_vertices, dtype=np.int32) % _N_CLASSES
        _labels_cache[key] = wp.array(labels, dtype=wp.int32, device=bench_case.device)
    return _labels_cache[key]


def _image_wp(bench_case: BenchCase) -> odt.Array3dFloat32:
    """Rasterize a float texture to sample back, built once outside the timed region."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _image_cache:
        _image_cache[key] = od.texture.rasterize_attribute(
            _uv_wp(bench_case), bench_case.faces_wp, _attribute_wp(bench_case), _RESOLUTION
        )
    return _image_cache[key]


def _class_image_wp(bench_case: BenchCase) -> odt.Array2dInt32:
    """Rasterize a class image to sample back, built once outside the timed region."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _class_image_cache:
        _class_image_cache[key] = od.texture.rasterize_discrete_attribute(
            _uv_wp(bench_case), bench_case.faces_wp, _labels_wp(bench_case), _RESOLUTION
        )
    return _class_image_cache[key]


@pytest.mark.benchmark(group="rasterize_attribute")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("resolution", _RESOLUTIONS)
def test_rasterize_attribute(bench_case: BenchCase, resolution: int) -> None:
    """Barycentric scatter of a 3-channel attribute, at 512^2 and 2048^2."""
    uv, faces = _uv_wp(bench_case), bench_case.faces_wp
    attribute = _attribute_wp(bench_case)
    image = bench_case.run(lambda: od.texture.rasterize_attribute(uv, faces, attribute, resolution))
    assert image.shape[:2] == (resolution, resolution)


@pytest.mark.benchmark(group="rasterize_discrete_attribute")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("resolution", _RESOLUTIONS)
def test_rasterize_discrete_attribute(bench_case: BenchCase, resolution: int) -> None:
    """The same scatter, resolving a per-pixel label argmax instead of interpolating."""
    uv, faces = _uv_wp(bench_case), bench_case.faces_wp
    labels = _labels_wp(bench_case)
    class_image = bench_case.run(
        lambda: od.texture.rasterize_discrete_attribute(uv, faces, labels, resolution)
    )
    assert class_image.shape == (resolution, resolution)


@pytest.mark.benchmark(group="remap_attribute_from_uv_linear")
@pytest.mark.benchlibs("ordito")
def test_remap_attribute_from_uv_linear(bench_case: BenchCase) -> None:
    """Per-vertex bilinear texture fetch — the ``wp.lerp`` gather path."""
    skip_larger_than(bench_case, "happy_buddha", "one 1024^2 source texture per mesh is cached")
    uv, image = _uv_wp(bench_case), _image_wp(bench_case)
    values = bench_case.run(lambda: od.texture.remap_attribute_from_uv(uv, image, order=1))
    assert values.shape[0] == bench_case.n_vertices


@pytest.mark.benchmark(group="remap_attribute_from_uv_nearest")
@pytest.mark.benchlibs("ordito")
def test_remap_attribute_from_uv_nearest(bench_case: BenchCase) -> None:
    """The same gather without the blend: isolates the bilinear interpolation cost."""
    skip_larger_than(bench_case, "happy_buddha", "one 1024^2 source texture per mesh is cached")
    uv, image = _uv_wp(bench_case), _image_wp(bench_case)
    values = bench_case.run(lambda: od.texture.remap_attribute_from_uv(uv, image, order=0))
    assert values.shape[0] == bench_case.n_vertices


@pytest.mark.benchmark(group="remap_discrete_attribute_from_uv")
@pytest.mark.benchlibs("ordito")
def test_remap_discrete_attribute_from_uv(bench_case: BenchCase) -> None:
    """Nearest-neighbor label fetch, preserving the ``-1`` uncovered sentinel."""
    skip_larger_than(bench_case, "happy_buddha", "one 1024^2 class image per mesh is cached")
    uv, class_image = _uv_wp(bench_case), _class_image_wp(bench_case)
    labels = bench_case.run(lambda: od.texture.remap_discrete_attribute_from_uv(uv, class_image))
    assert labels.shape[0] == bench_case.n_vertices
