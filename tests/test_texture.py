"""
Regression tests for ``ordito.texture``.

The forward rasterizers are compared against a vendored moderngl (OpenGL) reference, and
``rasterize_attribute`` also against open3d's vertex-attribute bake; the inverse
samplers against ``scipy.ndimage.map_coordinates``. OpenGL's top-left edge fill rule and a
barycentric pixel-centre coverage test part only on a pixel centre lying exactly on a rim edge,
which the fixture avoids, so the coverage masks are compared exactly; interpolated values carry
OpenGL's sub-pixel snapping and are compared at a measured bound.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable, Iterator
from typing import Literal

import moderngl
import numpy as np
import numpy.typing as npt
import open3d as o3d
import pytest
import warp as wp
from scipy.ndimage import map_coordinates

import ordito as od
import ordito.typing as odt
from tests.conftest import sphere_irregular_arrays
from tests.conversions import points_to_warp_uv, warp_empty


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
    color_tex = ctx.texture((resolution, resolution), 4, dtype="f4")
    out_frame_buffer = ctx.framebuffer(color_attachments=[color_tex])

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


def _rasterize_face_indices_gl(
    ctx: moderngl.Context,
    uv: npt.NDArray[np.float32],
    faces: npt.NDArray[np.int32],
    origin: tuple[float, float],
    pixel_size: float,
    shape: tuple[int, int],
    face_priority: npt.NDArray[np.float32] | None,
) -> npt.NDArray[np.int64]:
    """
    OpenGL reference: the face covering every pixel of a UV window, ``-1`` where none does.

    The faces are drawn unindexed so ``gl_PrimitiveID`` is the face index, with depth carrying
    the priority (spread over ``[-0.9, 0.9]``) under a ``<`` depth test, so the lowest priority
    wins and the first face drawn between equals. The window is mapped to clip space in float64.
    OpenGL reads rows back bottom-up; they are flipped so row ``0`` is the window's top, as the
    function under test returns them.
    """
    rows, columns = shape
    window = np.array([columns, rows], dtype=np.float64) * pixel_size
    xy = (uv.astype(np.float64)[faces] - np.asarray(origin)) / window * 2.0 - 1.0
    n_faces = faces.shape[0]
    depth = np.zeros(n_faces)
    if face_priority is not None and np.ptp(face_priority) > 0:
        priority = face_priority.astype(np.float64)
        depth = (priority - priority.min()) / np.ptp(priority) * 1.8 - 0.9
    corners = np.concatenate([xy, np.broadcast_to(depth[:, None, None], (n_faces, 3, 1))], axis=-1)

    program = ctx.program(
        vertex_shader=textwrap.dedent(
            """
            #version 330 core
            in vec3 in_position;
            void main() { gl_Position = vec4(in_position, 1.0); }
            """
        ),
        fragment_shader=textwrap.dedent(
            """
            #version 330 core
            out float face;
            void main() { face = float(gl_PrimitiveID + 1); }
            """
        ),
    )
    position_buffer = ctx.buffer(corners.astype(np.float32).tobytes())
    face_texture = ctx.texture((columns, rows), 1, dtype="f4")
    depth_texture = ctx.depth_texture((columns, rows))
    frame_buffer = ctx.framebuffer(color_attachments=[face_texture], depth_attachment=depth_texture)
    vertex_array = ctx.vertex_array(program, [(position_buffer, "3f", "in_position")])
    try:
        with ctx.scope(frame_buffer, enable_only=moderngl.DEPTH_TEST):
            ctx.depth_func = "<"
            frame_buffer.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
            vertex_array.render(moderngl.TRIANGLES)
            raw = face_texture.read()
    finally:
        vertex_array.release()
        frame_buffer.release()
        depth_texture.release()
        face_texture.release()
        position_buffer.release()
        program.release()
    image = np.flipud(np.frombuffer(raw, dtype=np.float32).reshape(rows, columns))
    return np.rint(image).astype(np.int64) - 1


def _remap_attribute_scipy(
    uv: npt.NDArray[np.float32], image: npt.NDArray[np.float32], order: Literal[0, 1]
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
def gl_context() -> Iterator[moderngl.Context]:
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
        # moderngl's stub types ``backend`` as a settings dict; the runtime takes the backend name.
        ctx = moderngl.create_context(standalone=True, backend="egl")  # pyright: ignore[reportArgumentType]
    except Exception as exc:  # noqa: BLE001 - any GL/EGL init failure should skip, not error
        pytest.skip(f"no EGL OpenGL context available: {exc!r}")
    yield ctx
    ctx.release()


def _gl_subpixel_bits(ctx: moderngl.Context) -> int | None:
    """
    Return the sub-pixel bits ``ctx``'s renderer snaps vertices to before interpolating.

    Measured on the anisotropic lattice below, both resolutions: NVIDIA's driver interpolates at
    vertices snapped to 8 bits, rounding to nearest (ordito snapped at 8 bits within 6e-07, at 6 or
    10 bits 1.8e-03 / 3.4e-04); Mesa's llvmpipe (CI's renderer) interpolates at the unsnapped
    positions (ordito unsnapped within 1.3e-06, at 8 bits 3.2e-04). ``None`` is no snapping. An
    unmeasured renderer skips, naming itself, rather than guessing a rule.
    """
    renderer = ctx.info["GL_RENDERER"]
    if "NVIDIA" in renderer:
        return 8
    if "llvmpipe" in renderer:
        return None
    pytest.skip(f"no measured sub-pixel snapping rule for the OpenGL renderer {renderer!r}")


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


def _anisotropic_grid_uv_mesh() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """``_grid_uv_mesh(8)`` squeezed to ``v`` in ``[0.135, 0.765]``: no symmetry under transpose."""
    uv_np, faces_np = _grid_uv_mesh(n=8)
    squeeze = np.array([1.0, 0.7], dtype=np.float32)
    lift = np.array([0.0, 0.1], dtype=np.float32)
    return (uv_np * squeeze + lift).astype(np.float32), faces_np


def _layered_uv_mesh() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32], int]:
    """
    Two copies of ``_anisotropic_grid_uv_mesh``, the second shifted by ``(0.21, 0.13)``.

    They overlap over about three quarters of the second, so the overlap's owner is decided by
    the face index or the priority. Returns the UVs, the ``(n_faces, 3)`` faces and the first
    face of the second copy.
    """
    uv_np, faces_np = _anisotropic_grid_uv_mesh()
    faces_np = faces_np.reshape(-1, 3)
    shifted = uv_np + np.array([0.21, 0.13], dtype=np.float32)
    layered_faces = np.concatenate([faces_np, faces_np + uv_np.shape[0]]).astype(np.int32)
    return np.concatenate([uv_np, shifted]).astype(np.float32), layered_faces, faces_np.shape[0]


def _edge_distances(
    uv_np: npt.NDArray[np.float32],
    faces_np: npt.NDArray[np.int32],
    face: npt.NDArray[np.int64],
    points: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Distance of each point to the nearest edge of its face, in UV units."""
    corners = uv_np.astype(np.float64)[faces_np[face]]  # (m, 3, 2)
    distances = np.full(points.shape[0], np.inf)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        start, edge = corners[:, a], corners[:, b] - corners[:, a]
        t = np.clip(((points - start) * edge).sum(1) / (edge * edge).sum(1), 0.0, 1.0)
        distance = np.linalg.norm(points - (start + t[:, None] * edge), axis=1)
        distances = np.minimum(distances, distance)
    return distances


def _interior_mask(n: int) -> npt.NDArray[np.bool_]:
    """Lattice vertices not on the outer ring (surrounded by covered pixels)."""
    rows, cols = np.divmod(np.arange(n * n), n)
    return (rows > 0) & (rows < n - 1) & (cols > 0) & (cols < n - 1)


def _upload_uv_mesh(
    uv_np: npt.NDArray[np.float32], faces_np: npt.NDArray[np.int32], device: str
) -> tuple[wp.array[wp.vec2], wp.array[wp.int32]]:
    """Upload a UV lattice and its flat face buffer."""
    return points_to_warp_uv(uv_np, device), wp.array(faces_np, dtype=wp.int32, device=device)


def _float_rows(values_np: npt.NDArray[np.float32], device: str) -> odt.Array2dFloat32:
    """Upload a ``(rows, channels)`` float32 attribute table."""
    return odt.as_array2d(wp.array(values_np, dtype=wp.float32, device=device), wp.float32)


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
@pytest.mark.parametrize(("resolution", "unsnapped_atol"), [(64, 1e-3), (256, 2.5e-4)])
def test_rasterize_attribute_matches_opengl(
    gl_context: moderngl.Context, device: str, resolution: int, unsnapped_atol: float
):
    """
    Class A at the renderer's own snapping: OpenGL's image, coverage and values.

    A renderer may snap every vertex to a sub-pixel grid before it interpolates. NVIDIA's driver
    keeps 8 bits, rounding to nearest; Mesa's llvmpipe (CI's renderer) interpolates unsnapped
    (``_gl_subpixel_bits``). Snapped the renderer's way, ordito's image equals it at the standard
    tolerance (measured 3.9e-07 in a NumPy transcription, both resolutions). Against NVIDIA's
    snapped image the unsnapped default carries that position error times the field's gradient,
    measured max **3.03e-04** at 64 px and **6.84e-05** at 256 px (attributes in ``[1, 2]``),
    which the second half still bounds, at ``unsnapped_atol`` (3.3x and 3.7x), so the default stays
    pinned.

    The coverage masks are compared with ``array_equal``. OpenGL's top-left fill rule and ordito's
    lowest-face owner only part ways on a pixel centre lying exactly on a *boundary* edge; on a
    shared interior edge both cover the pixel, and the interpolated field is continuous there. This
    fixture puts no pixel centre on its rim, so the masks are equal -- measured, on both devices and
    at both resolutions, snapped or not.

    The fixture is a lattice squeezed in ``v`` (``_anisotropic_grid_uv_mesh``) on purpose: the
    square lattice is symmetric under transposition, so a transposed mask scored a perfect
    agreement there and the obvious mutation was vacuous. Here a **transposed** mask disagrees on
    1 440 pixels at 64 px (22 218 at 256 px), so the ``array_equal`` bites on a swapped ``u`` /
    ``v`` axis or a missing vertical flip.

    A random Delaunay UV mesh was measured and rejected as the fixture: its slivers put a pixel
    centre within the snapping error of a rim edge (one coverage pixel apart at 64 px) and raise the
    value error to 2.9e-03 through their gradients, so the test would get looser, not stricter.

    The bug class this excludes: a wrong barycentric interpolation, a channel permutation, an
    off-by-one in the pixel-centre offset, a UV-to-NDC sign error or axis swap, a snapping scale
    off by a factor of two (on NVIDIA's driver 6 or 10 bits read 1.8e-03 and 3.4e-04 at 64 px), and
    any coverage rule
    that differs on a single pixel.
    """
    n_channels = 5
    rng = np.random.default_rng(20240704 + n_channels)
    uv_np, faces_np = _anisotropic_grid_uv_mesh()
    # Offset into [1, 2] so covered pixels are never ~0 (distinguishable from uncovered 0).
    attribute_np = (rng.random((uv_np.shape[0], n_channels), dtype=np.float32) + 1.0).astype(
        np.float32
    )

    image_gl = _rasterize_attribute_gl(
        gl_context, uv_np, faces_np.reshape(-1, 3), attribute_np, resolution
    )

    uv_wp, faces_wp = _upload_uv_mesh(uv_np, faces_np, device)
    attribute_wp = _float_rows(attribute_np, device)
    snapped_wp = od.texture.rasterize_attribute(
        uv_wp, faces_wp, attribute_wp, resolution, subpixel_bits=_gl_subpixel_bits(gl_context)
    ).numpy()
    image_wp = od.texture.rasterize_attribute(uv_wp, faces_wp, attribute_wp, resolution).numpy()

    covered_gl = image_gl[:, :, 0] > 0.5
    # Non-vacuous: both covered and uncovered pixels exist, and the mask is not transpose-symmetric.
    assert 0.2 < covered_gl.mean() < 0.9
    assert not np.array_equal(covered_gl, covered_gl.T)
    assert np.array_equal(snapped_wp[:, :, 0] > 0.5, covered_gl)
    assert np.allclose(snapped_wp, image_gl, rtol=1e-5, atol=1e-5)
    assert np.array_equal(image_wp[:, :, 0] > 0.5, covered_gl)
    assert np.allclose(image_wp[covered_gl], image_gl[covered_gl], rtol=0.0, atol=unsnapped_atol)


@pytest.mark.parity("rasterize_attribute", "open3d")
@pytest.mark.parametrize("resolution", [64, 256])
def test_rasterize_attribute_matches_open3d(device: str, resolution: int) -> None:
    """
    Class B (ordito's coverage): open3d's ``bake_vertex_attr_textures`` on the texels ordito covers.

    open3d bakes a vertex attribute into a texture through per-corner ``texture_uvs``, in ordito's
    image orientation (row ``0`` at ``v = 1``) and with the same barycentric interpolation at pixel
    centres. Its coverage is the named difference: ``margin`` dilates each UV island by that many
    pixels, and at ``margin=0`` it drops pixel centres on interior triangle edges (99 of ordito's
    2 320 texels at 64, some 19 pixels inside the island), so it is called at ``margin=1``. Then
    its texels cover every one of ordito's and its extra texels lie within a pixel of them. On
    ordito's texels the values agree to 3.6e-07 at 64 and 4.8e-07 at 256 on both devices. The
    fixture is the anisotropic UV lattice, which a transposed or flipped image would not match.
    **Mutation probe**: flipping the reference's rows leaves a maximum difference of 1.98 on
    ordito's texels (the flipped image's fill included).

    Non-vacuous: the attribute is random per vertex, so the shared texels are not constant.
    """
    uv_np, faces_np = _anisotropic_grid_uv_mesh()
    triangles_np = faces_np.reshape(-1, 3)
    attribute_np = np.random.default_rng(0).random((uv_np.shape[0], 3)).astype(np.float32)

    mesh_o3d = o3d.t.geometry.TriangleMesh()
    mesh_o3d.vertex.positions = o3d.core.Tensor(  # pyright: ignore[reportCallIssue]  # the stub drops dtype/device defaults
        np.column_stack([uv_np, np.zeros(uv_np.shape[0])]).astype(np.float32)
    )
    mesh_o3d.triangle.indices = o3d.core.Tensor(triangles_np.astype(np.int64))  # pyright: ignore[reportCallIssue]
    mesh_o3d.vertex["attribute"] = o3d.core.Tensor(attribute_np)  # pyright: ignore[reportCallIssue]
    mesh_o3d.triangle.texture_uvs = o3d.core.Tensor(uv_np[triangles_np])  # pyright: ignore[reportCallIssue]
    image_o3d = (
        mesh_o3d.bake_vertex_attr_textures(
            resolution, {"attribute"}, margin=1.0, fill=-1.0, update_material=False
        )["attribute"]
        .numpy()
        .astype(np.float32)
    )
    covered_o3d = (image_o3d != -1.0).all(axis=-1)

    image_np = od.texture.rasterize_attribute(
        *_upload_uv_mesh(uv_np, faces_np, device), _float_rows(attribute_np, device), resolution
    ).numpy()
    covered_np = (image_np != 0.0).any(axis=-1)

    assert covered_np.sum() > resolution  # non-vacuity: a real island
    assert np.ptp(image_o3d[covered_np]) > 0.5
    assert not (covered_np & ~covered_o3d).any()
    assert np.allclose(image_np[covered_np], image_o3d[covered_np], rtol=1e-5, atol=1e-5)
    # open3d's extra texels are its one-pixel margin: each touches a texel ordito covers.
    extra_np = covered_o3d & ~covered_np
    grown_np = covered_np.copy()
    grown_np[1:] |= covered_np[:-1]
    grown_np[:-1] |= covered_np[1:]
    grown_np[:, 1:] |= covered_np[:, :-1]
    grown_np[:, :-1] |= covered_np[:, 1:]
    grown_np[1:, 1:] |= covered_np[:-1, :-1]
    grown_np[:-1, :-1] |= covered_np[1:, 1:]
    grown_np[1:, :-1] |= covered_np[:-1, 1:]
    grown_np[:-1, 1:] |= covered_np[1:, :-1]
    assert not (extra_np & ~grown_np).any()


@pytest.mark.parity(
    "rasterize_discrete_attribute",
    "moderngl",
    benchmarked=False,
    reason="same decline as the continuous rasterizer above: an OpenGL row prices driver dispatch "
    "and an FBO round-trip rather than the argmax-over-accumulated-weights this group performs. It "
    "also needs a one-hot encoding on the reference side, so the two are not even doing the same "
    "amount of work -- the label agreement is what is comparable, and that is what this checks.",
)
@pytest.mark.parametrize("resolution", [64, 256])
def test_rasterize_discrete_attribute_matches_opengl(
    gl_context: moderngl.Context, device: str, resolution: int
):
    """
    Class B (equal coverage; equal labels outside a named argmax-tie band), via one-hot.

    Snapped as the renderer snaps (``_gl_subpixel_bits``, see the continuous test above) the labels
    are equal at every covered pixel but exact ties, which the reference breaks by its own rounding;
    the default, unsnapped, is what the rest of this docstring describes.

    OpenGL has no per-pixel argmax, so the reference rasterizes the labels **one-hot** as a
    continuous attribute and takes the argmax afterwards. Interpolating a one-hot vector and taking
    the argmax is barycentric nearest-label, which is what ordito's weight accumulation computes,
    and both break an exact tie towards the lowest label (``np.argmax`` takes the first maximum).

    So the labels are compared with ``array_equal`` everywhere except on the **tie band**: pixels
    where OpenGL's two largest interpolated weights are within ``1e-3`` of each other. There the
    argmax is decided by the reference's own error, not by a convention: its sub-pixel snapping
    (see the continuous test above) moves a weight by up to **2.7e-04** (measured, the largest gap
    at any disagreeing pixel over both resolutions and devices; on the square lattice the
    disagreements were exact ties at 3e-08 float noise). The band holds 0.04 % of the covered
    pixels at 64 px and 0.07 % at 256 px, and its size is asserted, so it cannot swallow the
    comparison.

    Mutation probes: a transposed class image fails the coverage ``array_equal`` (the fixture is
    the anisotropic lattice of the continuous test), and breaking ties towards the *highest* label
    in the reference leaves the off-band labels equal -- correctly, since ties live in the band --
    while shuffling the labels over the covered pixels drops off-band agreement to 0.27-0.28,
    chance for four labels.

    The bug class this excludes: a label gathered from the wrong vertex, an argmax over the wrong
    axis, any coverage disagreement, and the ``-1`` uncovered sentinel leaking into the covered
    region.
    """
    rng = np.random.default_rng(99)
    uv_np, faces_np = _anisotropic_grid_uv_mesh()
    labels_np = rng.integers(0, 4, size=uv_np.shape[0]).astype(np.int32)

    n_classes = int(labels_np.max() + 1)
    one_hot = np.eye(n_classes, dtype=np.float32)[labels_np]
    one_hot_image = _rasterize_attribute_gl(
        gl_context, uv_np, faces_np.reshape(-1, 3), one_hot, resolution
    )
    covered_gl = ~np.all(np.isclose(one_hot_image, 0.0), axis=-1)
    class_gl = np.argmax(one_hot_image, axis=-1).astype(np.int32)
    class_gl[~covered_gl] = -1
    top_two = np.sort(one_hot_image, axis=-1)[:, :, -2:]
    decided = covered_gl & (top_two[:, :, 1] - top_two[:, :, 0] >= 1e-3)

    uv_wp, faces_wp = _upload_uv_mesh(uv_np, faces_np, device)
    labels_wp = wp.array(labels_np, dtype=wp.int32, device=device)
    class_wp = od.texture.rasterize_discrete_attribute(
        uv_wp, faces_wp, labels_wp, resolution
    ).numpy()

    assert np.array_equal(class_wp >= 0, covered_gl)
    # Non-vacuous: the tie band is a thin minority and every label is present off it.
    assert decided.sum() > 0.99 * covered_gl.sum()
    assert np.unique(class_gl[decided]).shape[0] == n_classes
    assert np.array_equal(class_wp[decided], class_gl[decided])

    snapped_wp = od.texture.rasterize_discrete_attribute(
        uv_wp, faces_wp, labels_wp, resolution, subpixel_bits=_gl_subpixel_bits(gl_context)
    ).numpy()
    tied = covered_gl & (top_two[:, :, 1] - top_two[:, :, 0] < 1e-5)
    assert np.array_equal(snapped_wp >= 0, covered_gl)
    assert np.array_equal(snapped_wp[covered_gl & ~tied], class_gl[covered_gl & ~tied])


@pytest.mark.parity(
    "rasterize_face_indices",
    "moderngl",
    benchmarked=False,
    reason="same decline as the attribute rasterizers above: an OpenGL row prices driver dispatch, "
    "a depth-tested FBO and a readback rather than an algorithm. benchmarks/test_texture.py "
    "carries that decline.",
)
@pytest.mark.parametrize("prioritized", [False, True], ids=["index", "priority"])
@pytest.mark.parametrize(
    ("origin", "pixel_size", "shape"),
    [((0.1, 0.05), 0.009, (90, 120)), ((0.1, 0.05), 0.00227, (360, 480))],
)
def test_rasterize_face_indices_matches_opengl(
    gl_context: moderngl.Context,
    device: str,
    prioritized: bool,
    origin: tuple[float, float],
    pixel_size: float,
    shape: tuple[int, int],
) -> None:
    """
    Class B: OpenGL's face image, equal outside the driver's sub-pixel snapping band of an edge.

    The fixture is two overlapping copies of the anisotropic lattice (``_layered_uv_mesh``) seen
    through a window that starts off the origin, is not square and runs past both, so the image
    holds uncovered pixels, pixels of either copy alone, and an overlap decided by the face index
    (the first copy) or by priority (the second, of lower priority). Both sides take the lowest
    ``(priority, face index)`` covering a pixel centre: OpenGL by its ``<`` depth test in draw
    order.

    They part only on a pixel centre lying on an edge shared by two faces: OpenGL snaps the
    vertices to 1/256 of a pixel (8 bits, ``_gl_subpixel_bits``) and gives such a centre to one
    face by its fill rule, ordito tests it unsnapped against both. So every disagreeing pixel must
    lie within ``2 ** -8`` pixel of an edge of *both* answers' faces, and at most 20 may. Measured:
    no disagreement at 90 x 120; 9 at 360 x 480 by face index, all on the lattice's diagonals at
    1.8e-4 pixel, and none by priority; coverage equal everywhere, on both devices.

    Mutation probes at 360 x 480 (142 297 covered pixels, 105 205 of them the second copy's):
    rows flipped 157 706 pixels apart, the priority ignored or reversed 66 880, the window moved by
    half a pixel in ``u`` 2 494. The bug class this excludes: a flipped or transposed window, a
    pixel-centre offset, a wrong window scale, priorities ignored or compared the wrong way, and a
    tie broken towards the highest face.
    """
    uv_np, faces_np, second = _layered_uv_mesh()
    priority_np = (
        (np.arange(faces_np.shape[0]) < second).astype(np.float32) if prioritized else None
    )
    face_gl = _rasterize_face_indices_gl(
        gl_context, uv_np, faces_np, origin, pixel_size, shape, priority_np
    )

    uv_wp, faces_wp = _upload_uv_mesh(uv_np, faces_np.reshape(-1), device)
    priority_wp = None if priority_np is None else wp.array(priority_np, device=device)
    face_wp = od.texture.rasterize_face_indices(
        uv_wp, faces_wp, origin, pixel_size, shape, face_priority=priority_wp
    ).numpy()

    # Non-vacuous: uncovered pixels, both copies, and the overlap's owner set by the rule.
    assert 0.2 < (face_gl >= 0).mean() < 0.95
    assert (face_gl >= second).sum() > 0
    assert ((face_gl >= 0) & (face_gl < second)).sum() > 0
    assert np.array_equal(face_wp >= 0, face_gl >= 0)
    rows, columns = np.nonzero(face_wp != face_gl)
    assert rows.size <= 20
    # Pixel (i, j)'s centre: row 0 is the window's top.
    flipped_rows = shape[0] - 1 - rows
    centres = np.asarray(origin, dtype=np.float64) + (
        np.stack([columns, flipped_rows], axis=1).astype(np.float64) + 0.5
    ) * np.float64(pixel_size)
    for face in (face_gl[rows, columns], face_wp[rows, columns].astype(np.int64)):
        assert np.all(_edge_distances(uv_np, faces_np, face, centres) < pixel_size * 2.0**-8)


@pytest.mark.parametrize("resolution", [64, 256])
def test_rasterize_face_indices_shares_rasterize_attribute_frame(device: str, resolution: int):
    """
    Ordito against ordito: over ``[0, 1]`` the face image covers what ``rasterize_attribute`` does.

    Pixel for pixel, row ``0`` at ``v = 1`` in both. The anisotropic lattice puts no pixel centre
    within rounding of its rim, so the two frames
    (``float64`` for the window, ``float32`` for the texture) cover the same pixels; its coverage
    is not symmetric under a flip or a transpose, so either breaks the equality (OpenGL is the
    oracle for both functions in the tests above; this pins their agreement with each other).
    """
    uv_np, faces_np = _anisotropic_grid_uv_mesh()
    uv_wp, faces_wp = _upload_uv_mesh(uv_np, faces_np, device)
    ones_wp = _float_rows(np.ones((uv_np.shape[0], 1), dtype=np.float32), device)
    covered_np = od.texture.rasterize_attribute(uv_wp, faces_wp, ones_wp, resolution).numpy()[
        ..., 0
    ]
    face_np = od.texture.rasterize_face_indices(
        uv_wp, faces_wp, (0.0, 0.0), 1.0 / resolution, (resolution, resolution)
    ).numpy()
    covered = covered_np > 0.5
    assert not np.array_equal(covered, np.flipud(covered))
    assert np.array_equal(face_np >= 0, covered)


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
@pytest.mark.parametrize("n_channels", [None, 3], ids=["rank2_image", "3ch"])
def test_remap_attribute_matches_scipy(device: str, order: Literal[0, 1], n_channels: int | None):
    """
    Class A against ``scipy.ndimage.map_coordinates``, on both interpolation orders.

    One test covers two benchmark groups because ``order`` is exactly what separates them:
    ``order=1`` is ``remap_attribute_from_uv_linear`` and ``order=0`` is ``..._nearest``, so both
    markers sit here rather than one of them borrowing the other's oracle.

    ``n_channels=None`` is a bare ``(height, width)`` image, a separate branch of the wrapper: it
    returns a ``(n, 1)`` column as the one-channel image does, and the reference helper promotes
    it with ``image[:, :, None]``.

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
    rng = np.random.default_rng(7 * order + (n_channels or 0))
    height, width = 40, 48
    image_shape = (height, width) if n_channels is None else (height, width, n_channels)
    image_np = rng.random(image_shape, dtype=np.float32)
    uv_np = rng.random((200, 2), dtype=np.float32)

    values_np = _remap_attribute_scipy(uv_np, image_np, order)

    uv_wp = points_to_warp_uv(uv_np, device)
    image_wp = (
        _float_rows(image_np, device)
        if n_channels is None
        else odt.as_array3d(wp.array(image_np, dtype=wp.float32, device=device), wp.float32)
    )
    values_wp = od.texture.remap_attribute_from_uv(uv_wp, image_wp, order=order).numpy()

    assert values_wp.shape == (200, n_channels or 1)
    assert np.allclose(values_wp, values_np, rtol=1e-4, atol=1e-4)


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
    class_image_wp = odt.as_array2d(
        wp.array(class_image_np, dtype=wp.int32, device=device), wp.int32
    )
    labels_wp = od.texture.remap_discrete_attribute_from_uv(uv_wp, class_image_wp).numpy()

    # Non-vacuity: a constant answer would satisfy array_equal against a constant reference.
    assert np.unique(labels_np).size == n_labels
    assert np.array_equal(labels_wp, labels_np)


# --------------------------------------------------------------------------------------------
# Round-trip identity
# --------------------------------------------------------------------------------------------
def test_rasterize_remap_roundtrip(device: str):
    """
    Not a library comparison: rasterizing and sampling back recovers the vertex values.

    A globally linear field is recovered by bilinear sampling at interior vertices to the pixel
    grid's resolution, and a per-vertex label exactly by the discrete pair.
    """
    rng = np.random.default_rng(11)
    n = 10
    resolution = 128
    uv_np, faces_np = _grid_uv_mesh(n=n)
    uv_wp, faces_wp = _upload_uv_mesh(uv_np, faces_np, device)
    interior = _interior_mask(n)

    u = uv_np[:, 0]
    v = uv_np[:, 1]
    attribute_np = np.stack([u, v, u + v], axis=1).astype(np.float32)
    attribute_wp = _float_rows(attribute_np, device)
    image = od.texture.rasterize_attribute(uv_wp, faces_wp, attribute_wp, resolution)
    recovered = od.texture.remap_attribute_from_uv(uv_wp, image, order=1).numpy()
    assert np.allclose(recovered[interior], attribute_np[interior], rtol=1e-2, atol=1e-2)

    labels_np = rng.integers(0, 5, size=uv_np.shape[0]).astype(np.int32)
    labels_wp = wp.array(labels_np, dtype=wp.int32, device=device)
    class_image = od.texture.rasterize_discrete_attribute(uv_wp, faces_wp, labels_wp, resolution)
    recovered_labels = od.texture.remap_discrete_attribute_from_uv(uv_wp, class_image).numpy()
    assert np.array_equal(recovered_labels[interior], labels_np[interior])


# --------------------------------------------------------------------------------------------
# Independence from the owner pass's split and from the device
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("slices", [1, 8])
def test_rasterize_ignores_split_and_device(
    device: str, slices: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Ordito against ordito: the images are the same bits whatever the owner pass's split or device.

    No library rasterizes with this coverage rule. ``sphere_irregular`` projected onto ``xy``
    stacks its front and back (the lowest-face rule decides every covered pixel) and turns its
    rim faces edge-on, whose pixel centres sit within rounding of an edge: the pixels a contracted
    (FMA) coverage test settles by loop shape and by device. The default split is compared with
    one forced split per parameter, and both with the CPU's answer. The face image takes each
    face's mean ``z`` as its priority, so the front and back stack through the priority keys.
    """
    vertices_np, faces_np = sphere_irregular_arrays()
    xy = vertices_np[:, :2]
    uv_np = ((xy - xy.min(axis=0)) / np.ptp(xy, axis=0)).astype(np.float32)
    faces_flat_np = faces_np.reshape(-1).astype(np.int32)
    attribute_np = vertices_np.astype(np.float32)
    labels_np = (np.arange(uv_np.shape[0]) % 5).astype(np.int32)
    resolution = 512

    priority_np = vertices_np[faces_np, 2].mean(axis=1).astype(np.float32)

    def images(
        on: str,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        uv_wp, faces_wp = _upload_uv_mesh(uv_np, faces_flat_np, on)
        labels_wp = wp.array(labels_np, dtype=wp.int32, device=on)
        image = od.texture.rasterize_attribute(
            uv_wp, faces_wp, _float_rows(attribute_np, on), resolution
        )
        class_image = od.texture.rasterize_discrete_attribute(
            uv_wp, faces_wp, labels_wp, resolution
        )
        face_image = od.texture.rasterize_face_indices(
            uv_wp,
            faces_wp,
            (0.0, 0.0),
            1.0 / resolution,
            (resolution, resolution),
            face_priority=wp.array(priority_np, device=on),
        )
        return image.numpy(), class_image.numpy(), face_image.numpy()

    image_np, class_np, face_np = images(device)
    monkeypatch.setattr(od.texture, "_owner_slices", lambda *_: slices)
    image_split_np, class_split_np, face_split_np = images(device)
    monkeypatch.undo()
    image_cpu_np, class_cpu_np, face_cpu_np = images("cpu")
    assert 0 < (class_np >= 0).sum() < class_np.size
    assert np.array_equal(image_split_np, image_np)
    assert np.array_equal(class_split_np, class_np)
    assert np.array_equal(face_split_np, face_np)
    assert np.array_equal(image_cpu_np, image_np)
    assert np.array_equal(class_cpu_np, class_np)
    assert np.array_equal(face_cpu_np, face_np)


@pytest.mark.parametrize(
    ("priority", "owner"),
    [
        pytest.param(None, 0, id="none"),
        pytest.param([1.0, 0.0], 1, id="lower_wins"),
        pytest.param([2.0, 2.0], 0, id="tie_lowest_index"),
        pytest.param([0.0, -0.0], 0, id="signed_zeros_tie"),
        pytest.param([float("nan"), 5.0], 1, id="nan_loses"),
        pytest.param([float("inf"), float("nan")], 0, id="nan_below_inf"),
    ],
)
def test_rasterize_face_indices_priority_rule(
    device: str, priority: list[float] | None, owner: int
) -> None:
    """
    Not a library comparison: the tie rules of ``face_priority``, which OpenGL cannot state.

    OpenGL's depth test compares a quantized depth, and every ``NaN`` fails it. Two copies of one
    triangle cover the same pixels, so the rule alone picks the owner of every
    covered pixel. A face with a non-finite corner beside them covers nothing.
    """
    uv_wp = _point_uv(device, [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [float("nan"), 0.5]])
    faces_wp = wp.array([0, 1, 2, 0, 1, 2, 0, 1, 3], dtype=wp.int32, device=device)
    priority_wp = None
    if priority is not None:
        priority_wp = wp.array([*priority, -1.0], dtype=wp.float32, device=device)
    face_np = od.texture.rasterize_face_indices(
        uv_wp, faces_wp, (0.0, 0.0), 0.125, (8, 8), face_priority=priority_wp
    ).numpy()
    covered = face_np >= 0
    assert 0 < covered.sum() < covered.size
    assert np.all(face_np[covered] == owner)


# --------------------------------------------------------------------------------------------
# NaN-UV handling
# --------------------------------------------------------------------------------------------
def test_nan_uv_rows(device: str):
    uv_np = np.array([[0.5, 0.5], [np.nan, np.nan], [0.25, 0.75]], dtype=np.float32)
    image_np = np.random.default_rng(0).random((16, 16, 2), dtype=np.float32)

    uv_wp = points_to_warp_uv(uv_np, device)
    image_wp = odt.as_array3d(wp.array(image_np, dtype=wp.float32, device=device), wp.float32)
    values = od.texture.remap_attribute_from_uv(uv_wp, image_wp, order=1).numpy()
    assert np.all(np.isnan(values[1]))
    assert not np.any(np.isnan(values[0]))
    assert not np.any(np.isnan(values[2]))

    class_image_np = np.random.default_rng(1).integers(0, 3, size=(16, 16)).astype(np.int32)
    class_image_wp = odt.as_array2d(
        wp.array(class_image_np, dtype=wp.int32, device=device), wp.int32
    )
    labels = od.texture.remap_discrete_attribute_from_uv(uv_wp, class_image_wp).numpy()
    assert labels[1] == -1
    assert labels[0] >= 0
    assert labels[2] >= 0


# --------------------------------------------------------------------------------------------
# Edge cases / validation
# --------------------------------------------------------------------------------------------
def test_empty_faces(device: str):
    uv_wp = wp.array(np.zeros((0, 2), dtype=np.float32), dtype=wp.vec2, device=device)
    faces_wp = wp.array(np.zeros(0, dtype=np.int32), dtype=wp.int32, device=device)
    attribute_wp = odt.as_array2d(
        wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.float32, device=device), wp.float32
    )
    image = od.texture.rasterize_attribute(uv_wp, faces_wp, attribute_wp, 8).numpy()
    assert image.shape == (8, 8, 3)
    assert np.all(image == 0.0)

    labels_wp = wp.array(np.zeros(0, dtype=np.int32), dtype=wp.int32, device=device)
    class_image = od.texture.rasterize_discrete_attribute(uv_wp, faces_wp, labels_wp, 8).numpy()
    assert class_image.shape == (8, 8)
    assert np.all(class_image == -1)

    face_image = od.texture.rasterize_face_indices(uv_wp, faces_wp, (0.0, 0.0), 0.5, (3, 5))
    assert face_image.shape == (3, 5)
    assert np.all(face_image.numpy() == -1)


def _point_uv(device: str, rows: list[list[float]]) -> wp.array[wp.vec2]:
    return points_to_warp_uv(np.array(rows, dtype=np.float32), device)


def _no_faces(device: str) -> wp.array[wp.int32]:
    return wp.zeros(0, dtype=wp.int32, device=device)


def _one_face(device: str) -> wp.array[wp.int32]:
    return wp.array([0, 1, 2], dtype=wp.int32, device=device)


_THREE_UV = [[0.5, 0.5], [0.2, 0.2], [0.8, 0.8]]
_OUT_OF_RANGE_UV = [[0.5, 0.5], [1.5, 0.2]]


def _face_indices(
    device: str,
    origin: tuple[float, float] = (0.0, 0.0),
    pixel_size: float = 0.1,
    shape: tuple[int, int] = (4, 4),
) -> odt.Array2dInt32:
    """``rasterize_face_indices`` of one face over a window, one argument varied by the caller."""
    return od.texture.rasterize_face_indices(
        _point_uv(device, _THREE_UV), _one_face(device), origin, pixel_size, shape
    )


_VALIDATION_CASES: list[tuple[str, Callable[[str], object], str]] = [
    (
        "rasterize_zero_resolution",
        lambda device: od.texture.rasterize_attribute(
            _point_uv(device, [[0.5, 0.5]]),
            _no_faces(device),
            _float_rows(np.zeros((1, 1), dtype=np.float32), device),
            0,
        ),
        "Resolution must be positive",
    ),
    (
        "rasterize_negative_subpixel_bits",
        lambda device: od.texture.rasterize_attribute(
            _point_uv(device, _THREE_UV),
            _one_face(device),
            _float_rows(np.zeros((3, 1), dtype=np.float32), device),
            8,
            subpixel_bits=-1,
        ),
        "subpixel_bits must be non-negative",
    ),
    (
        "rasterize_uv_out_of_range",
        lambda device: od.texture.rasterize_attribute(
            _point_uv(device, _OUT_OF_RANGE_UV),
            _no_faces(device),
            _float_rows(np.zeros((2, 1), dtype=np.float32), device),
            8,
        ),
        r"\[0, 1\]",
    ),
    (
        "rasterize_discrete_negative_label",
        lambda device: od.texture.rasterize_discrete_attribute(
            _point_uv(device, _THREE_UV),
            _one_face(device),
            wp.array([0, -1, 2], dtype=wp.int32, device=device),
            8,
        ),
        "greater than or equal to 0",
    ),
    (
        "rasterize_row_count_mismatch",
        lambda device: od.texture.rasterize_attribute(
            _point_uv(device, _THREE_UV),
            _one_face(device),
            _float_rows(np.zeros((2, 1), dtype=np.float32), device),
            8,
        ),
        "row count mismatch",
    ),
    (
        "rasterize_discrete_row_count_mismatch",
        lambda device: od.texture.rasterize_discrete_attribute(
            _point_uv(device, _THREE_UV),
            _one_face(device),
            wp.zeros(2, dtype=wp.int32, device=device),
            8,
        ),
        "row count mismatch",
    ),
    *(
        (f"face_indices_{case_id}", call, match)
        for case_id, call, match in (
            ("no_rows", lambda device: _face_indices(device, shape=(0, 4)), "shape must hold"),
            ("no_columns", lambda device: _face_indices(device, shape=(4, -1)), "shape must hold"),
            (
                "zero_pixel_size",
                lambda device: _face_indices(device, pixel_size=0.0),
                "pixel_size must be positive",
            ),
            (
                "nan_pixel_size",
                lambda device: _face_indices(device, pixel_size=float("nan")),
                "pixel_size must be positive",
            ),
            (
                "inf_origin",
                lambda device: _face_indices(device, origin=(0.0, float("inf"))),
                "origin must be finite",
            ),
        )
    ),
    (
        "face_indices_priority_count",
        lambda device: od.texture.rasterize_face_indices(
            _point_uv(device, _THREE_UV),
            _one_face(device),
            (0.0, 0.0),
            0.1,
            (4, 4),
            face_priority=wp.zeros(2, dtype=wp.float32, device=device),
        ),
        "one value a face",
    ),
    (
        "remap_uv_out_of_range",
        lambda device: od.texture.remap_attribute_from_uv(
            _point_uv(device, _OUT_OF_RANGE_UV),
            odt.as_array2d(wp.zeros((4, 4), dtype=wp.float32, device=device), wp.float32),
        ),
        r"\[0, 1\]",
    ),
    (
        "remap_discrete_uv_out_of_range",
        lambda device: od.texture.remap_discrete_attribute_from_uv(
            _point_uv(device, _OUT_OF_RANGE_UV),
            odt.as_array2d(wp.zeros((4, 4), dtype=wp.int32, device=device), wp.int32),
        ),
        r"\[0, 1\]",
    ),
    *(
        (
            f"remap_order_{order}",
            lambda device, order=order: od.texture.remap_attribute_from_uv(
                _point_uv(device, [[0.25, 0.25], [0.6, 0.7]]),
                odt.as_array2d(wp.zeros((4, 4), dtype=wp.float32, device=device), wp.float32),
                order=order,  # pyright: ignore[reportArgumentType]  # the off-menu value under test
            ),
            "order must be 0",
        )
        for order in (2, -1, 0.5)
    ),
]


@pytest.mark.parametrize(
    ("call", "match"),
    [pytest.param(call, match, id=case_id) for case_id, call, match in _VALIDATION_CASES],
)
def test_rejects_invalid_arguments(device: str, call: Callable[[str], object], match: str) -> None:
    """
    Not a library comparison: argument-validation guards, which no reference library exposes.

    Each documented ``ValueError`` of the four entry points. The row-count guard matters more than
    its message: without it the rasterizing kernel indexes the shorter buffer out of range, which
    on the cpu device is host-heap corruption rather than a wrong answer.
    ``remap_discrete_attribute_from_uv`` reaches its ``[0, 1]`` guard only *indirectly*, through
    its ``remap_attribute_from_uv`` call, so its documented ``Raises`` entry is a claim about a
    guard it does not itself run -- the kind that goes stale unnoticed. ``order`` was read as
    ``SAMPLE_BILINEAR if order == 1 else SAMPLE_NEAREST``, so every value other than ``1`` sampled
    nearest-neighbour and returned a plausible image -- the failure mode a ``Literal`` annotation
    hides from a caller whose value arrives through a variable; both documented orders still run
    against their own references in ``test_remap_attribute_matches_scipy``.
    """
    with pytest.raises(ValueError, match=match):
        call(device)


def test_remap_attribute_from_uv_rejects_an_image_of_rank_four(device: str) -> None:
    """Not a library comparison: the documented ``TypeError`` for an image of neither rank."""
    with pytest.raises(TypeError, match=r"image must be \(H, W\) or \(H, W, C\)"):
        od.texture.remap_attribute_from_uv(
            _point_uv(device, [[0.25, 0.25]]),
            wp.zeros((2, 2, 1, 1), dtype=wp.float32, device=device),  # pyright: ignore[reportArgumentType]  # the rank under test
        )


def _blocky_texels(rng: np.random.Generator, blocks: int, block: int) -> npt.NDArray[np.uint8]:
    """Return a ``(blocks * block, blocks * block, 4)`` RGBA texture constant over each block."""
    colors = rng.integers(0, 256, size=(blocks, blocks, 4), dtype=np.uint8)
    return np.repeat(np.repeat(colors, block, axis=0), block, axis=1)


@pytest.mark.parity("material_colors", "trimesh")
def test_material_colors_matches_trimesh(device: str) -> None:
    """
    Class A against trimesh's ``PBRMaterial.to_color``, where the two pixel conventions agree.

    trimesh reads the nearest texel under ``col = u * (W - 1)``, ordito blends four at pixel centres
    (``col = u * W - 0.5``), so they part by up to half a texel's blend wherever the texture varies.
    The texture here is constant over 8x8 blocks and each UV lies within half a texel of a block's
    centre, where both read the block's colour exactly; the copies shifted by whole tiles check the
    wrap, which trimesh applies modulo its texel count. Swapping the vertical axis (``row = v * H``)
    or dropping the wrap fails on value: the blocks are random and distinct. The factor is checked
    with no image, where trimesh returns ``baseColorFactor``.
    """
    from PIL import Image
    from trimesh.visual.material import PBRMaterial

    rng = np.random.default_rng(0)
    blocks, block = 8, 8
    texels_np = _blocky_texels(rng, blocks, block)
    centres = (np.arange(blocks) + 0.5) / blocks
    uv_np = np.stack(np.meshgrid(centres, centres), axis=-1).reshape(-1, 2)
    uv_np += rng.uniform(-0.5, 0.5, size=uv_np.shape) / (blocks * block)
    shifts_np = np.array([[0.0, 0.0], [1.0, 0.0], [-2.0, 3.0]])
    uv_np = (uv_np[None] + shifts_np[:, None]).reshape(-1, 2).astype(np.float32)

    colors_tm = PBRMaterial(baseColorTexture=Image.fromarray(texels_np)).to_color(uv_np)
    assert np.unique(colors_tm, axis=0).shape[0] > blocks  # not one colour
    image_wp = wp.array(texels_np.astype(np.float32) / 255.0, dtype=wp.float32, device=device)
    material = od.texture.Material(image=odt.as_array3d(image_wp, wp.float32))
    colors_wp = od.texture.material_colors(material, points_to_warp_uv(uv_np, device))
    assert np.allclose(colors_wp.numpy(), colors_tm / 255.0, rtol=1e-5, atol=1e-6)

    factor_np = np.array([0.2, 0.4, 0.6, 1.0], dtype=np.float32)
    plain_tm = PBRMaterial(baseColorFactor=factor_np).to_color(uv_np)
    plain = od.texture.Material(base_color=wp.vec4(*factor_np.tolist()))
    plain_wp = od.texture.material_colors(plain, points_to_warp_uv(uv_np, device))
    assert np.allclose(
        plain_wp.numpy(), np.broadcast_to(plain_tm / 255.0, (uv_np.shape[0], 4)), atol=0.5 / 255.0
    )


def test_material_colors_blends_tiles_and_scales(device: str) -> None:
    """
    Not a library comparison: the bilinear blend, the tiling and the factor, on four texels.

    No reference blends a material's texels (trimesh reads the nearest one): at a texel centre the
    colour is that texel times the factor, half-way between two it is their mean, half-way across
    the edge it is the mean of the first and last texel (the wrap; a clamped sampler reads the edge
    texel), and an Inf or NaN UV reads NaN.
    """
    texels_np = np.array(
        [[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
        dtype=np.float32,
    )
    image_wp = wp.array(texels_np, dtype=wp.float32, device=device)
    material = od.texture.Material(
        image=odt.as_array3d(image_wp, wp.float32), base_color=wp.vec4(0.5, 1.0, 1.0, 1.0)
    )
    uv_np = np.array(
        [[0.125, 0.5], [0.25, 0.5], [0.0, 0.5], [1.125, -3.5], [np.nan, 0.5], [np.inf, 0.5]],
        dtype=np.float32,
    )
    colors_np = od.texture.material_colors(material, points_to_warp_uv(uv_np, device)).numpy()
    factor_np = np.array([0.5, 1.0, 1.0, 1.0], dtype=np.float32)
    assert np.allclose(colors_np[0], texels_np[0, 0] * factor_np)
    assert np.allclose(colors_np[1], (texels_np[0, 0] + texels_np[0, 1]) / 2 * factor_np)
    assert np.allclose(colors_np[2], (texels_np[0, 0] + texels_np[0, 3]) / 2 * factor_np)
    assert np.allclose(colors_np[3], colors_np[0])
    assert np.isnan(colors_np[4:]).all()
    empty_wp = od.texture.material_colors(material, warp_empty(0, wp.vec2, device))
    assert empty_wp.shape == (0,)


@pytest.mark.parametrize(
    ("image", "error", "match"),
    [
        pytest.param(((2, 2), wp.float32), TypeError, "ndim", id="rank"),
        pytest.param(((2, 2, 4), wp.float64), TypeError, "dtype", id="dtype"),
        pytest.param(((2, 2, 3), wp.float32), ValueError, "4 \\(RGBA\\) channels", id="channels"),
        pytest.param(((0, 2, 4), wp.float32), ValueError, "a texel", id="empty"),
    ],
)
def test_material_rejects_an_invalid_image(
    device: str, image: tuple[tuple[int, ...], type], error: type[Exception], match: str
) -> None:
    """Not a library comparison: a material's image is a non-empty ``(H, W, 4)`` ``float32``."""
    shape, dtype = image
    with pytest.raises(error, match=match):
        od.texture.Material(image=wp.zeros(shape, dtype=dtype, device=device))  # pyright: ignore[reportArgumentType]  # the invalid image under test


def test_material_colors_rejects_an_image_on_another_device() -> None:
    """Not a library comparison: the UVs and the image must share a device."""
    if not wp.is_cuda_available():
        pytest.skip("needs a CUDA device beside the CPU")
    image_wp = wp.ones((1, 1, 4), dtype=wp.float32, device="cpu")
    material = od.texture.Material(image=odt.as_array3d(image_wp, wp.float32))
    with pytest.raises(RuntimeError):
        od.texture.material_colors(material, wp.zeros(1, dtype=wp.vec2, device="cuda:0"))
