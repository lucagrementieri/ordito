"""
Regression tests for ``triwarp.texture``.

The forward rasterizers are compared against a vendored moderngl (OpenGL) reference; the inverse
samplers against ``scipy.ndimage.map_coordinates``. Because OpenGL's top-left edge fill rule
differs subtly from a barycentric pixel-center coverage test, comparisons of the covered/uncovered
mask allow a thin boundary band of disagreement, while interpolated values are compared only where
both rasterizers cover the pixel (the interpolated field is continuous, so values agree there).
"""

from __future__ import annotations

import textwrap

import moderngl
import numpy as np
import numpy.typing as npt
import pytest
import warp as wp
from scipy.ndimage import map_coordinates

import triwarp as tw
from tests.conversions import points_to_warp_uv


# --------------------------------------------------------------------------------------------
# Reference implementations
# --------------------------------------------------------------------------------------------
def _rasterize_attribute_gl(
    ctx: moderngl.Context,
    uv: npt.NDArray[np.float32],
    faces: npt.NDArray[np.int32],
    attribute: npt.NDArray[np.float32],
    resolution: int,
) -> npt.NDArray[np.float32]:
    """OpenGL reference: interpolate a per-vertex attribute into a UV texture (chunked vec3s)."""
    n_vertices, n_channels = attribute.shape
    n_chunks = (n_channels + 2) // 3
    padded = np.zeros((n_vertices, n_chunks * 3), dtype=np.float32)
    padded[:, :n_channels] = attribute
    chunks_3d = padded.reshape(n_vertices, n_chunks, 3)

    vertex_shader = textwrap.dedent(
        """
        #version 330 core
        in vec2 in_uv;
        in vec3 in_attr;
        out vec3 v_attr;
        void main() {
            gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0);
            v_attr = in_attr;
        }
        """
    )
    fragment_shader = textwrap.dedent(
        """
        #version 330 core
        in vec3 v_attr;
        out vec4 fragColor;
        void main() { fragColor = vec4(v_attr, 1.0); }
        """
    )
    program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
    uv_buffer = ctx.buffer(uv.astype(np.float32).tobytes())
    face_buffer = ctx.buffer(faces.astype(np.int32).tobytes())
    out_frame_buffer = ctx.framebuffer(
        color_attachments=[ctx.texture((resolution, resolution), 4, dtype="f4")]
    )
    color_tex = out_frame_buffer.color_attachments[0]

    out_parts: list[npt.NDArray[np.float32]] = []
    try:
        for i in range(n_chunks):
            start = i * 3
            n_keep = min(3, n_channels - start)
            attribute_buffer = ctx.buffer(chunks_3d[:, i, :].tobytes())
            out_vertex_array = ctx.vertex_array(
                program,
                [(uv_buffer, "2f", "in_uv"), (attribute_buffer, "3f", "in_attr")],
                index_buffer=face_buffer,
                index_element_size=4,
            )
            try:
                out_frame_buffer.use()
                ctx.clear(0.0, 0.0, 0.0, 1.0)
                out_vertex_array.render(moderngl.TRIANGLES)
                raw = color_tex.read()
                plane = np.frombuffer(raw, dtype=np.float32).reshape(resolution, resolution, 4)
                out_parts.append(np.flipud(plane)[:, :, :n_keep])
            finally:
                out_vertex_array.release()
                attribute_buffer.release()
    finally:
        uv_buffer.release()
        face_buffer.release()
        color_tex.release()
        out_frame_buffer.release()
        program.release()
    return np.concatenate(out_parts, axis=-1) if len(out_parts) > 1 else out_parts[0].copy()


def _remap_attribute_scipy(
    uv: npt.NDArray[np.float32], image: npt.NDArray[np.float32], order: int
) -> npt.NDArray[np.float32]:
    """Scipy reference for the pixel-center inverse sampling."""
    image_hwc = image[:, :, None] if image.ndim == 2 else image
    height, width, n_channels = image_hwc.shape
    u = uv[:, 0].astype(np.float64)
    v = uv[:, 1].astype(np.float64)
    rows = (1.0 - v) * height - 0.5
    cols = u * width - 0.5
    out = np.empty((uv.shape[0], n_channels), dtype=np.float32)
    for c in range(n_channels):
        out[:, c] = map_coordinates(
            image_hwc[:, :, c].astype(np.float64),
            [rows, cols],
            order=order,
            mode="nearest",
            prefilter=False,
        )
    return out


# --------------------------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gl_context():
    """
    Yield a standalone EGL context, or skip naming the *driver* -- never the missing package.

    ``moderngl`` is a hard ``test``-group dependency like ``igl`` and ``pymeshlab``, so it is
    imported plainly at module scope and a missing wheel is a failure, not a skip. What genuinely
    cannot be declared in ``pyproject.toml`` is a working EGL driver, and that is the only thing
    this skip is allowed to be about: the module previously opened with
    ``pytest.importorskip("moderngl")``, which would have deleted all 354 lines of coverage silently
    if the import ever broke.
    """
    try:
        ctx = moderngl.create_context(standalone=True, backend="egl")
    except Exception as exc:  # noqa: BLE001 - any GL/EGL init failure should skip, not error
        pytest.skip(f"no EGL OpenGL context available: {exc!r}")
    yield ctx
    ctx.release()


def _grid_uv_mesh(
    n: int, pad: float = 0.05
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """Triangulated ``n x n`` lattice covering ``[pad, 1 - pad]^2`` in UV space."""
    coords = np.linspace(pad, 1.0 - pad, n)
    uu, vv = np.meshgrid(coords, coords, indexing="xy")
    uv = np.stack([uu.ravel(), vv.ravel()], axis=1).astype(np.float32)
    faces: list[int] = []
    for r in range(n - 1):
        for c in range(n - 1):
            i = r * n + c
            faces += [i, i + 1, i + n, i + 1, i + n + 1, i + n]
    return uv, np.array(faces, dtype=np.int32)


def _interior_mask(n: int) -> npt.NDArray[np.bool_]:
    """Lattice vertices not on the outer ring (surrounded by covered pixels)."""
    rows, cols = np.divmod(np.arange(n * n), n)
    return (rows > 0) & (rows < n - 1) & (cols > 0) & (cols < n - 1)


# --------------------------------------------------------------------------------------------
# Forward rasterization vs OpenGL
# --------------------------------------------------------------------------------------------
@pytest.mark.parity(
    "rasterize_attribute",
    "moderngl",
    benchmarked=False,
    reason="an OpenGL rasterization prices driver dispatch, an FBO allocation and a context "
    "round-trip rather than an algorithm, so a timed row would compare a GPU driver against a Warp "
    "kernel and read as a throughput number for neither. benchmarks/test_texture.py carries that "
    "decline; moderngl is registered as a test-only reference so this claim is still checked.",
)
@pytest.mark.parametrize("n_channels", [1, 2, 5])
def test_rasterize_attribute_matches_opengl(gl_context, device: str, n_channels: int):
    """
    Class C (a coverage fraction plus values on the jointly-covered pixels).

    No correspondence is available pixel-for-pixel: OpenGL's top-left fill rule and triwarp's
    barycentric pixel-centre coverage test disagree by construction on a boundary pixel, so the
    comparison is a coverage *agreement fraction* and then an exact value check restricted to the
    pixels both rasterizers claim. The interpolated field is continuous, which is what makes the
    restriction sound rather than convenient -- where both cover, both are interpolating the same
    affine function over the same triangle.

    Measured at ``n_channels=5``: coverage agreement is **1.000000** -- the fill-rule band this
    fixture could exercise is empty -- with 82.1 % of the image covered, 3 364 jointly-covered
    pixels and a max value difference of **1.99e-04** against the ``1e-3`` bound. So the ``0.98``
    threshold is *headroom for the documented fill-rule difference, not a fitted number*; it has
    never been approached on this fixture.

    Mutation probe, and it has a trap worth recording: **transposing** triwarp's coverage mask
    also scores **1.000000**, because ``_grid_uv_mesh`` is symmetric under transposition -- so the
    obvious mutation is vacuous here and would "prove" the bound bites when it does not. A
    **shuffle** is the valid probe: it drops agreement to **0.708**, i.e. 29.2 % disagreement
    against the 2 % the
    threshold allows, a **14.6x** margin.

    The bug class this excludes: a wrong barycentric interpolation, a channel permutation, an
    off-by-one in the pixel-centre offset, a UV-to-NDC sign error, and any coverage rule wrong by
    more than a boundary pixel. What it does *not* see is a systematic half-pixel shift small enough
    to keep the same pixel set, which the round-trip tests below are what cover.
    """
    rng = np.random.default_rng(20240704 + n_channels)
    uv_np, faces_np = _grid_uv_mesh(n=8)
    # Offset into [1, 2] so covered pixels are never ~0 (distinguishable from uncovered 0).
    attribute_np = (rng.random((uv_np.shape[0], n_channels), dtype=np.float32) + 1.0).astype(
        np.float32
    )
    resolution = 64

    image_gl = _rasterize_attribute_gl(
        gl_context, uv_np, faces_np.reshape(-1, 3), attribute_np, resolution
    )

    uv_wp = points_to_warp_uv(uv_np, device)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)
    attribute_wp = wp.array(attribute_np, dtype=wp.float32, device=device)
    image_wp = tw.texture.rasterize_attribute(uv_wp, faces_wp, attribute_wp, resolution).numpy()

    covered_gl = image_gl[:, :, 0] > 0.5
    covered_wp = image_wp[:, :, 0] > 0.5
    # Coverage masks agree except for a thin boundary band (fill-rule differences).
    assert np.mean(covered_gl == covered_wp) > 0.98
    both = covered_gl & covered_wp
    assert both.sum() > 0
    assert np.allclose(image_wp[both], image_gl[both], rtol=1e-3, atol=1e-3)


@pytest.mark.parity(
    "rasterize_discrete_attribute",
    "moderngl",
    benchmarked=False,
    reason="same decline as the continuous rasterizer above: an OpenGL row prices driver dispatch "
    "and an FBO round-trip rather than the argmax-over-accumulated-weights this group performs. It "
    "also needs a one-hot encoding on the reference side, so the two are not even doing the same "
    "amount of work -- the label agreement is what is comparable, and that is what this checks.",
)
def test_rasterize_discrete_attribute_matches_opengl(gl_context, device: str):
    """
    Class C (coverage fraction plus label agreement), through a named one-hot transform.

    OpenGL has no per-pixel argmax, so the reference rasterizes the labels **one-hot** as a
    continuous attribute and takes the argmax afterwards. That is the transform, and it is not free:
    interpolating a one-hot vector and then taking an argmax is barycentric nearest-label, which is
    what triwarp's weight accumulation computes -- but the two break argmax *ties* differently, so
    the label comparison is a fraction rather than an equality.

    Measured: coverage agreement **1.000000** and label agreement **0.999405** -- 2 disagreeing
    pixels of 3 364, both on a tie edge -- against the ``0.98`` bound.

    Mutation probe: shuffling the reference labels over the jointly-covered pixels drops agreement
    to **0.2815**, which is chance for this fixture's 4 labels. So the bound sits **34x** above
    chance
    in disagreement terms (0.06 % measured against 71.8 % shuffled, with 2 % allowed).

    The bug class this excludes: a label gathered from the wrong vertex, an argmax over the wrong
    axis, a coverage rule that disagrees by more than a boundary pixel, and the ``-1`` uncovered
    sentinel leaking into the covered region. Ties near an edge are the residual, and they are the
    reason this is a fraction and not ``array_equal``.
    """
    rng = np.random.default_rng(99)
    n = 8
    uv_np, faces_np = _grid_uv_mesh(n=n)
    labels_np = rng.integers(0, 4, size=uv_np.shape[0]).astype(np.int32)
    resolution = 64

    n_classes = int(labels_np.max() + 1)
    one_hot = np.eye(n_classes, dtype=np.float32)[labels_np]
    one_hot_image = _rasterize_attribute_gl(
        gl_context, uv_np, faces_np.reshape(-1, 3), one_hot, resolution
    )
    covered_gl = ~np.all(np.isclose(one_hot_image, 0.0), axis=-1)
    class_gl = np.argmax(one_hot_image, axis=-1).astype(np.int32)
    class_gl[~covered_gl] = -1

    uv_wp = points_to_warp_uv(uv_np, device)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)
    labels_wp = wp.array(labels_np, dtype=wp.int32, device=device)
    class_wp = tw.texture.rasterize_discrete_attribute(
        uv_wp, faces_wp, labels_wp, resolution
    ).numpy()

    covered_wp = class_wp >= 0
    assert np.mean(covered_gl == covered_wp) > 0.98
    both = covered_gl & covered_wp
    # Labels agree on the vast majority of jointly-covered pixels (argmax ties near edges differ).
    assert np.mean(class_wp[both] == class_gl[both]) > 0.98


# --------------------------------------------------------------------------------------------
# Inverse sampling vs scipy
# --------------------------------------------------------------------------------------------
@pytest.mark.parity(
    "remap_attribute_from_uv_linear",
    "scipy",
    benchmarked=False,
    reason="scipy.ndimage.map_coordinates samples a plain grid and knows nothing about UV space or "
    "mesh topology, so it is the right oracle for the interpolation and the wrong baseline for "
    "the operation -- a row would time a host grid sampler against a device gather over a "
    "parametrization. benchmarks/test_texture.py carries that decline.",
)
@pytest.mark.parity(
    "remap_attribute_from_uv_nearest",
    "scipy",
    benchmarked=False,
    reason="scipy.ndimage.map_coordinates samples a plain grid and knows nothing about UV space or "
    "mesh topology, so it is the right oracle for the interpolation and the wrong baseline for "
    "the operation -- a row would time a host grid sampler against a device gather over a "
    "parametrization. benchmarks/test_texture.py carries that decline.",
)
@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("n_channels", [1, 3])
def test_remap_attribute_matches_scipy(device: str, order: int, n_channels: int):
    """
    Class A against ``scipy.ndimage.map_coordinates``, on both interpolation orders.

    One test covers two benchmark groups because ``order`` is exactly what separates them:
    ``order=1`` is ``remap_attribute_from_uv_linear`` and ``order=0`` is ``..._nearest``, so both
    markers sit here rather than one of them borrowing the other's oracle.

    The reference has to be *constructed* to agree, and that construction is the content of the
    comparison: ``_remap_attribute_scipy`` maps UV to pixel coordinates as
    ``rows = (1 - v) * height - 0.5`` and ``cols = u * width - 0.5``, which is the pixel-*centre*
    convention with the v axis flipped, and passes ``mode="nearest"`` plus ``prefilter=False`` so
    that scipy clamps at the border and does no spline pre-filter. Any of those four wrong and the
    two
    disagree by a half-pixel or at the edges -- so this test pins the convention as much as the
    arithmetic.

    Measured max absolute difference against the ``1e-4`` bound: **exactly 0.0** at ``order=0`` for
    both channel counts (a nearest fetch is the same integer index on both sides, so there is
    nothing to round), and **1.37e-06** / **1.67e-06** at ``order=1``, which is float32 lerp
    against
    float64 spline evaluation. Mutation probe: shuffling the reference rows takes the max difference
    to **0.75-0.97**, the full range of the data, so the bound clears the mutation by four orders of
    magnitude.
    """
    rng = np.random.default_rng(7 * order + n_channels)
    height, width = 40, 48
    image_np = rng.random((height, width, n_channels), dtype=np.float32)
    uv_np = rng.random((200, 2), dtype=np.float32)

    values_np = _remap_attribute_scipy(uv_np, image_np, order)

    uv_wp = points_to_warp_uv(uv_np, device)
    image_wp = wp.array(image_np, dtype=wp.float32, device=device)
    values_wp = tw.texture.remap_attribute_from_uv(uv_wp, image_wp, order=order).numpy()

    assert values_wp.shape == (200, n_channels)
    assert np.allclose(values_wp, values_np, rtol=1e-4, atol=1e-4)


@pytest.mark.parity(
    "remap_attribute_from_uv_linear",
    "scipy",
    benchmarked=False,
    reason="scipy.ndimage.map_coordinates samples a plain grid and knows nothing about UV space or "
    "mesh topology, so it is the right oracle for the interpolation and the wrong baseline for "
    "the operation -- a row would time a host grid sampler against a device gather over a "
    "parametrization. benchmarks/test_texture.py carries that decline.",
)
def test_remap_attribute_2d_image(device: str):
    """
    Class A: the rank-2 image path, which the parametrized test above never reaches.

    ``remap_attribute_from_uv`` accepts a bare ``(height, width)`` image as well as
    ``(height, width, channels)`` and returns a ``(n, 1)`` column either way; the reference helper
    promotes the 2-D case with ``image[:, :, None]``. That promotion is the only difference from
    [`test_remap_attribute_matches_scipy`], and it is a real branch in the wrapper rather than a
    convenience, which is why it has its own comparison at ``order=1``.
    """
    rng = np.random.default_rng(3)
    image_np = rng.random((32, 32), dtype=np.float32)
    uv_np = rng.random((50, 2), dtype=np.float32)
    values_np = _remap_attribute_scipy(uv_np, image_np, order=1)

    uv_wp = points_to_warp_uv(uv_np, device)
    image_wp = wp.array(image_np, dtype=wp.float32, device=device)
    values_wp = tw.texture.remap_attribute_from_uv(uv_wp, image_wp, order=1).numpy()
    assert values_wp.shape == (50, 1)
    assert np.allclose(values_wp[:, 0], values_np[:, 0], rtol=1e-4, atol=1e-4)


@pytest.mark.parity(
    "remap_discrete_attribute_from_uv",
    "scipy",
    benchmarked=False,
    reason="same decline as the continuous samplers above: map_coordinates samples a plain "
    "grid with no notion of UV space, so it is the interpolation oracle and not a baseline "
    "for the operation. At order=0 it is additionally doing strictly less -- no label "
    "semantics, no -1 sentinel -- so a row would price a host nearest fetch against a "
    "device gather over a parametrization.",
)
@pytest.mark.parametrize("n_labels", [2, 5])
def test_remap_discrete_attribute_matches_scipy(device: str, n_labels: int) -> None:
    """
    Class A: a class image sampled back to labels is ``map_coordinates`` at ``order=0``.

    Nearest-neighbour sampling of an integer image is exactly what the discrete remapper does, and
    scipy performs it on the float64 promotion of the same array -- so the two must agree
    *exactly*, not to a tolerance: ``order=0`` picks a texel rather than blending it, and both sides
    address it through the identical pixel-centre convention
    (``rows = (1 - v) * height - 0.5``, ``cols = u * width - 0.5``, ``mode="nearest"``).

    Measured: **0 mismatches of 300** queries at 5 labels, with all 5 labels present in the answer.
    Mutation probe: shuffling the reference labels drops agreement to **0.180**, which is chance at
    5 labels, so ``array_equal`` clears the mutation completely rather than by a margin.

    Random UVs are what make the exactness claim safe. A tie -- a query landing exactly on a pixel
    boundary -- is where a rounding convention could separate the two, and it has measure zero here;
    the ``n_labels=2`` arm is the one that would surface a systematic off-by-one: with only
    two labels a half-pixel shift changes the answer on a large fraction of the image rather
    than on a thin band.

    The bug class this excludes: a flipped v axis, a half-pixel offset, a row/column transposition,
    and a label read through the continuous path (which would blend two labels into a third).
    """
    rng = np.random.default_rng(11 + n_labels)
    height, width = 40, 48
    class_image_np = rng.integers(0, n_labels, size=(height, width)).astype(np.int32)
    uv_np = rng.random((300, 2), dtype=np.float32)

    rows_np = (1.0 - uv_np[:, 1].astype(np.float64)) * height - 0.5
    columns_np = uv_np[:, 0].astype(np.float64) * width - 0.5
    labels_np = map_coordinates(
        class_image_np.astype(np.float64),
        [rows_np, columns_np],
        order=0,
        mode="nearest",
        prefilter=False,
    ).astype(np.int32)

    uv_wp = points_to_warp_uv(uv_np, device)
    class_image_wp = wp.array(class_image_np, dtype=wp.int32, device=device)
    labels_wp = tw.texture.remap_discrete_attribute_from_uv(uv_wp, class_image_wp).numpy()

    # Non-vacuity: a constant answer would satisfy array_equal against a constant reference.
    assert np.unique(labels_np).size == n_labels
    assert np.array_equal(labels_wp, labels_np)


# --------------------------------------------------------------------------------------------
# Round-trip identity
# --------------------------------------------------------------------------------------------
def test_rasterize_remap_roundtrip_linear(device: str):
    n = 10
    uv_np, faces_np = _grid_uv_mesh(n=n)
    # Globally-linear field -> bilinear sampling recovers it exactly at interior vertices.
    u = uv_np[:, 0]
    v = uv_np[:, 1]
    attribute_np = np.stack([u, v, u + v], axis=1).astype(np.float32)
    resolution = 128

    uv_wp = points_to_warp_uv(uv_np, device)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)
    attribute_wp = wp.array(attribute_np, dtype=wp.float32, device=device)
    image = tw.texture.rasterize_attribute(uv_wp, faces_wp, attribute_wp, resolution)
    recovered = tw.texture.remap_attribute_from_uv(uv_wp, image, order=1).numpy()

    interior = _interior_mask(n)
    assert np.allclose(recovered[interior], attribute_np[interior], rtol=1e-2, atol=1e-2)


def test_rasterize_remap_roundtrip_discrete(device: str):
    rng = np.random.default_rng(11)
    n = 10
    uv_np, faces_np = _grid_uv_mesh(n=n)
    labels_np = rng.integers(0, 5, size=uv_np.shape[0]).astype(np.int32)
    resolution = 128

    uv_wp = points_to_warp_uv(uv_np, device)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)
    labels_wp = wp.array(labels_np, dtype=wp.int32, device=device)
    class_image = tw.texture.rasterize_discrete_attribute(uv_wp, faces_wp, labels_wp, resolution)
    recovered = tw.texture.remap_discrete_attribute_from_uv(uv_wp, class_image).numpy()

    interior = _interior_mask(n)
    assert np.array_equal(recovered[interior], labels_np[interior])


# --------------------------------------------------------------------------------------------
# NaN-UV handling
# --------------------------------------------------------------------------------------------
def test_nan_uv_rows(device: str):
    uv_np = np.array([[0.5, 0.5], [np.nan, np.nan], [0.25, 0.75]], dtype=np.float32)
    image_np = np.random.default_rng(0).random((16, 16, 2), dtype=np.float32)

    uv_wp = points_to_warp_uv(uv_np, device)
    image_wp = wp.array(image_np, dtype=wp.float32, device=device)
    values = tw.texture.remap_attribute_from_uv(uv_wp, image_wp, order=1).numpy()
    assert np.all(np.isnan(values[1]))
    assert not np.any(np.isnan(values[0]))
    assert not np.any(np.isnan(values[2]))

    class_image_np = np.random.default_rng(1).integers(0, 3, size=(16, 16)).astype(np.int32)
    class_image_wp = wp.array(class_image_np, dtype=wp.int32, device=device)
    labels = tw.texture.remap_discrete_attribute_from_uv(uv_wp, class_image_wp).numpy()
    assert labels[1] == -1
    assert labels[0] >= 0
    assert labels[2] >= 0


# --------------------------------------------------------------------------------------------
# Edge cases / validation
# --------------------------------------------------------------------------------------------
def test_empty_faces(device: str):
    uv_wp = wp.array(np.zeros((0, 2), dtype=np.float32), dtype=wp.vec2, device=device)
    faces_wp = wp.array(np.zeros(0, dtype=np.int32), dtype=wp.int32, device=device)
    attribute_wp = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.float32, device=device)
    image = tw.texture.rasterize_attribute(uv_wp, faces_wp, attribute_wp, 8).numpy()
    assert image.shape == (8, 8, 3)
    assert np.all(image == 0.0)

    labels_wp = wp.array(np.zeros(0, dtype=np.int32), dtype=wp.int32, device=device)
    class_image = tw.texture.rasterize_discrete_attribute(uv_wp, faces_wp, labels_wp, 8).numpy()
    assert class_image.shape == (8, 8)
    assert np.all(class_image == -1)


def test_invalid_resolution(device: str):
    uv_wp = wp.array(np.array([[0.5, 0.5]], dtype=np.float32), dtype=wp.vec2, device=device)
    faces_wp = wp.array(np.zeros(0, dtype=np.int32), dtype=wp.int32, device=device)
    attribute_wp = wp.array(np.zeros((1, 1), dtype=np.float32), dtype=wp.float32, device=device)
    with pytest.raises(ValueError, match="Resolution must be positive"):
        tw.texture.rasterize_attribute(uv_wp, faces_wp, attribute_wp, 0)


def test_uv_out_of_range(device: str):
    uv_np = np.array([[0.5, 0.5], [1.5, 0.2]], dtype=np.float32)
    uv_wp = points_to_warp_uv(uv_np, device)
    faces_wp = wp.array(np.zeros(0, dtype=np.int32), dtype=wp.int32, device=device)
    attribute_wp = wp.array(np.zeros((2, 1), dtype=np.float32), dtype=wp.float32, device=device)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        tw.texture.rasterize_attribute(uv_wp, faces_wp, attribute_wp, 8)


def test_discrete_negative_labels(device: str):
    uv_np = np.array([[0.5, 0.5], [0.2, 0.2], [0.8, 0.8]], dtype=np.float32)
    uv_wp = points_to_warp_uv(uv_np, device)
    faces_wp = wp.array(np.array([0, 1, 2], dtype=np.int32), dtype=wp.int32, device=device)
    labels_wp = wp.array(np.array([0, -1, 2], dtype=np.int32), dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        tw.texture.rasterize_discrete_attribute(uv_wp, faces_wp, labels_wp, 8)


def test_rasterize_row_count_mismatch(device: str):
    """
    Not a library comparison: an argument-validation guard, which no reference library exposes.

    Both rasterizers document a ``ValueError`` when ``uv`` and ``attribute`` disagree on their row
    count. Without it the rasterizing kernel indexes the shorter buffer out of range, which on the
    cpu device is host-heap corruption rather than a wrong answer (CLAUDE.md section 12.1).
    """
    uv_np = np.array([[0.5, 0.5], [0.2, 0.2], [0.8, 0.8]], dtype=np.float32)
    uv_wp = points_to_warp_uv(uv_np, device)
    faces_wp = wp.array(np.array([0, 1, 2], dtype=np.int32), dtype=wp.int32, device=device)
    attribute_wp = wp.array(np.zeros((2, 1), dtype=np.float32), dtype=wp.float32, device=device)
    with pytest.raises(ValueError, match="row count mismatch"):
        tw.texture.rasterize_attribute(uv_wp, faces_wp, attribute_wp, 8)
    labels_wp = wp.array(np.zeros(2, dtype=np.int32), dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="row count mismatch"):
        tw.texture.rasterize_discrete_attribute(uv_wp, faces_wp, labels_wp, 8)


def test_remap_uv_out_of_range(device: str):
    """
    Not a library comparison: an argument-validation guard, which no reference library exposes.

    Both samplers document the same ``[0, 1]`` ``ValueError`` the rasterizers raise, and
    ``remap_discrete_attribute_from_uv`` reaches it only *indirectly*, through its
    ``remap_attribute_from_uv`` call -- so its documented ``Raises`` entry is a claim about a
    guard it does not itself run, which is exactly the kind that goes stale unnoticed.
    """
    uv_wp = points_to_warp_uv(np.array([[0.5, 0.5], [1.5, 0.2]], dtype=np.float32), device)
    image_wp = wp.zeros((4, 4), dtype=wp.float32, device=device)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        tw.texture.remap_attribute_from_uv(uv_wp, image_wp)
    class_image_wp = wp.zeros((4, 4), dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        tw.texture.remap_discrete_attribute_from_uv(uv_wp, class_image_wp)


def test_remap_rejects_an_off_menu_order(device: str):
    """
    Not a library comparison: the ``order`` menu's own guard.

    ``order`` was read as ``SAMPLE_BILINEAR if order == 1 else SAMPLE_NEAREST``, so every value
    other than ``1`` sampled nearest-neighbour and returned a plausible image -- the failure mode
    a ``Literal`` annotation hides from a caller whose value arrives through a variable rather
    than a literal. Both documented orders still run, and they must disagree on this image, or
    the test would pass against a function that ignored ``order`` entirely.
    """
    uv_wp = points_to_warp_uv(np.array([[0.25, 0.25], [0.6, 0.7]], dtype=np.float32), device)
    image_np = np.arange(16, dtype=np.float32).reshape(4, 4)
    image_wp = wp.array(image_np, dtype=wp.float32, device=device)
    for order in (2, -1, 0.5):
        with pytest.raises(ValueError, match="order must be 0"):
            tw.texture.remap_attribute_from_uv(uv_wp, image_wp, order=order)
    bilinear_np = tw.texture.remap_attribute_from_uv(uv_wp, image_wp, order=1).numpy()
    nearest_np = tw.texture.remap_attribute_from_uv(uv_wp, image_wp, order=0).numpy()
    assert np.isfinite(bilinear_np).all()
    assert np.isfinite(nearest_np).all()
    assert not np.allclose(bilinear_np, nearest_np)
