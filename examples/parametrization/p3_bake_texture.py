from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="P3",
    title="Baking a field into a texture and reading it back",
    summary="""
    With a UV chart, any per-vertex quantity can be stored as an image. Here the mean curvature
    of the bunny patch from [`principal_curvature`][ordito.curvature.principal_curvature] is
    baked by [`rasterize_attribute`][ordito.texture.rasterize_attribute], which fills every
    texel a triangle covers by interpolating across it in the [`lscm`][ordito.parametrization.lscm]
    chart. [`remap_attribute_from_uv`][ordito.texture.remap_attribute_from_uv] samples the image
    back onto the vertices: at 512 texels the round trip is close to exact, while a 32-texel
    texture keeps only the broad shape of the field.
    """,
    credits=(("trimesh: texture", "https://trimesh.org/examples.html"),),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("face_patch", device)
    boundary = od.boundary.longest_boundary_loop(vertices, faces).numpy()
    pins = wp.array([boundary[0], boundary[boundary.size // 2]], dtype=wp.int32, device=device)
    pinned_uv = wp.array([(0.0, 0.0), (1.0, 0.0)], dtype=wp.vec2, device=device)
    uv = od.parametrization.lscm(vertices, faces, pins, pinned_uv).numpy()
    uv = (uv - uv.min(axis=0)) / np.ptp(uv, axis=0).max() * 0.98 + 0.01  # fit into [0, 1]
    uv = wp.array(uv, dtype=wp.vec2, device=device)

    _, _, k1, k2 = od.curvature.principal_curvature(vertices, faces)
    mean = 0.5 * (k1.numpy() + k2.numpy())
    field = od.typing.as_array2d(
        wp.array(mean[:, None], dtype=wp.float32, device=device), wp.float32
    )

    images, restored = {}, {}
    for resolution in (512, 32):
        images[resolution] = od.texture.rasterize_attribute(uv, faces, field, resolution)
        restored[resolution] = od.texture.remap_attribute_from_uv(uv, images[resolution])
        error = np.abs(restored[resolution].numpy()[:, 0] - mean)
        relative = np.median(error) / np.ptp(mean)
        print(f"{resolution:>3} texels: median error {relative:.2%} of the field's range")
    # --8<-- [end:code]
    return {
        "mean": mean,
        "image": images[512].numpy()[:, :, 0],
        "restored": {k: v.numpy()[:, 0] for k, v in restored.items()},
        "mask": od.texture.rasterize_attribute(
            uv,
            faces,
            od.typing.as_array2d(
                wp.ones((mean.size, 1), dtype=wp.float32, device=device), wp.float32
            ),
            512,
        ).numpy()[:, :, 0],
    }


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("face_patch")
    clim = r.percentile_clim(result["mean"], 2, 98)

    def texture(ax: Any) -> None:
        image = np.where(result["mask"] > 0, result["image"], np.nan)
        ax.imshow(image, cmap="RdBu_r", vmin=clim[0], vmax=clim[1], extent=(0, 1, 0, 1))
        ax.set_xticks([])
        ax.set_yticks([])

    def surface(values: np.ndarray, title: str) -> r.Panel:
        return r.Panel(
            [r.Mesh(vertices, faces, scalars=values, cmap="RdBu_r", clim=clim)], title=title
        )

    return r.Figure(
        [
            surface(result["mean"], "Mean curvature"),
            r.Plot(texture, title="Baked texture (512 x 512)"),
            surface(result["restored"][512], "Read back from 512 texels"),
            surface(result["restored"][32], "Read back from 32 texels"),
        ],
        camera=data.camera("face_patch"),
        panel_size=(540, 560),
    )
