"""
Off-screen rendering of example figures: PyVista panels composed into one transparent PNG.

An example's ``figure`` returns a [`Figure`][] of panels. A 3-D panel is a list of layers (meshes,
polylines, points, arrows, voxel boxes) rendered off-screen by PyVista; a 2-D panel is a matplotlib
drawing callback. Panels are rendered one by one at a fixed size and laid out in a grid by
matplotlib, so titles and spacing are uniform across the gallery. The background is transparent, so
one image serves the light and the dark documentation theme; text is mid-grey for the same reason.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap
from PIL import Image

if TYPE_CHECKING:
    from pyvista.plotting._typing import ScalarBarArgs

mpl.use("Agg")

GREEN = "#76b900"
"""Default surface colour (the NVIDIA green the site's palette uses)."""

LIGHT_GREEN = "#b8dc80"
"""A lighter surface tint: the *input* side of a before/after pair, or a context surface."""

GREY = "#c8c8c8"
"""A neutral surface: geometry that is shown for context only."""

RED = "#e8412c"
"""Boundary rims, defects, outliers: anything the example points at."""

BLUE = "#2c7be8"
"""A second highlight colour (the other mesh of a pair, a second curve set)."""

ORANGE = "#f39c12"
"""A third highlight colour."""

MAGENTA = "#c2189b"
"""A fourth highlight colour (non-manifold elements)."""

TEXT = "#808080"
"""Text colour: mid-grey reads on both the light and the dark theme."""

PANEL_SIZE = (720, 600)
"""Pixel size of one rendered panel."""

pv.global_theme.font.color = TEXT
pv.global_theme.font.family = "arial"


# --------------------------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------------------------


@dataclass
class Mesh:
    """A triangle mesh, shaded in one colour or coloured by a per-vertex or per-face scalar."""

    vertices: np.ndarray
    faces: np.ndarray
    color: str = GREEN
    scalars: np.ndarray | None = None
    cmap: Any = "viridis"
    clim: tuple[float, float] | None = None
    scalar_bar: str | None = None
    show_edges: bool = False
    edge_color: str = "#303030"
    line_width: float = 0.6
    opacity: float = 1.0
    smooth: bool = True
    rgb: bool = False
    face_colors: np.ndarray | None = None


class MeshStyle(TypedDict, total=False):
    """Keyword arguments of ``Mesh`` shared by several layers of one figure (``**style``)."""

    show_edges: bool
    edge_color: str
    line_width: float
    opacity: float
    smooth: bool


@dataclass
class Lines:
    """Polylines drawn as tubes: rims, loops, paths, contours."""

    polylines: Sequence[np.ndarray]
    color: str = RED
    width: float = 5.0
    closed: bool = False
    scalars: Sequence[np.ndarray] | None = None
    cmap: Any = "viridis"


@dataclass
class Segments:
    """Independent line segments, given as an ``(m, 2, 3)`` array of endpoints."""

    segments: np.ndarray
    color: str = RED
    width: float = 3.0


@dataclass
class Points:
    """A point cloud drawn as shaded spheres."""

    points: np.ndarray
    color: str = GREEN
    scalars: np.ndarray | None = None
    cmap: Any = "viridis"
    clim: tuple[float, float] | None = None
    scalar_bar: str | None = None
    size: float = 6.0
    opacity: float = 1.0
    rgb: bool = False


@dataclass
class Arrows:
    """Vectors drawn as arrow glyphs at ``origins``."""

    origins: np.ndarray
    vectors: np.ndarray
    color: str = BLUE
    scale: float = 1.0


@dataclass
class Spheres:
    """Balls of individual radii (inscribed spheres, query balls)."""

    centers: np.ndarray
    radii: np.ndarray
    color: str = BLUE
    opacity: float = 0.5


@dataclass
class Boxes:
    """Axis-aligned cubes of side ``size`` centred at ``centers`` (voxels)."""

    centers: np.ndarray
    size: float
    color: str = GREEN
    scalars: np.ndarray | None = None
    cmap: Any = "viridis"
    show_edges: bool = True
    opacity: float = 1.0


@dataclass
class WireBox:
    """An oriented box given by its eight corners in ``itertools.product`` order."""

    corners: np.ndarray
    color: str = BLUE
    width: float = 3.0


@dataclass
class Plane:
    """A square plane patch through ``center`` with normal ``normal``."""

    center: np.ndarray
    normal: np.ndarray
    size: float
    color: str = BLUE
    opacity: float = 0.35


Layer = Mesh | Lines | Segments | Points | Arrows | Spheres | Boxes | WireBox | Plane


@dataclass
class Camera:
    """A view direction (from the target towards the eye), an up vector and a zoom factor."""

    direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    zoom: float = 1.0
    parallel: bool = False


@dataclass
class Panel:
    """One 3-D view of a list of layers."""

    layers: Sequence[Layer]
    title: str = ""
    camera: Camera | None = None
    bounds: np.ndarray | None = None
    """Bounds ``(2, 3)`` that frame the view; defaults to the first layer's (or the figure's)."""


@dataclass
class Plot:
    """One 2-D panel drawn by a matplotlib callback ``draw(ax)``."""

    draw: Callable[[Any], None]
    title: str = ""
    aspect: str = "equal"


@dataclass
class Table:
    """A small text table, drawn as a panel."""

    rows: Sequence[Sequence[str]]
    title: str = ""
    header: bool = True


@dataclass
class Figure:
    """The panels of one example image, laid out left to right, ``ncols`` per row."""

    panels: Sequence[Panel | Plot | Table]
    ncols: int | None = None
    camera: Camera = field(default_factory=Camera)
    link_bounds: bool = True
    """Frame every 3-D panel with the first panel's bounds, so before/after views line up."""
    panel_size: tuple[int, int] = PANEL_SIZE


# --------------------------------------------------------------------------------------------
# Colour maps
# --------------------------------------------------------------------------------------------


def striped(cmap: str = "viridis", bands: int = 24, dark: float = 0.72) -> ListedColormap:
    """Return a colour map whose alternate bands are darkened (the libigl geodesic look)."""
    base = plt.get_cmap(cmap)(np.linspace(0.0, 1.0, 1024))
    band = (np.arange(1024) * bands // 1024) % 2 == 1
    base[band, :3] *= dark
    return ListedColormap(base)


def symmetric_clim(values: np.ndarray, percentile: float = 98.0) -> tuple[float, float]:
    """Return a colour range centred at zero that clips the tails at ``percentile``."""
    bound = float(np.percentile(np.abs(values[np.isfinite(values)]), percentile))
    bound = bound if bound > 0.0 else 1.0
    return (-bound, bound)


def percentile_clim(
    values: np.ndarray, low: float = 2.0, high: float = 98.0
) -> tuple[float, float]:
    """Return a colour range clipping both tails at the given percentiles."""
    finite = values[np.isfinite(values)]
    lo, hi = float(np.percentile(finite, low)), float(np.percentile(finite, high))
    return (lo, hi if hi > lo else lo + 1.0)


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------


def polydata(vertices: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    """Return a PyVista surface from ``(n, 3)`` vertices and ``(m, 3)`` or flat faces."""
    tri = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    cells = np.hstack([np.full((tri.shape[0], 1), 3, dtype=np.int64), tri]).ravel()
    return pv.PolyData(np.asarray(vertices, dtype=np.float64), cells)


def _polyline(points: np.ndarray, closed: bool) -> pv.PolyData:
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    ids = list(range(n)) + ([0] if closed and n > 2 else [])
    return pv.PolyData(points, lines=np.array([len(ids), *ids], dtype=np.int64))


def _layer_bounds(layer: Layer) -> np.ndarray | None:
    if isinstance(layer, Mesh):
        pts = np.asarray(layer.vertices)
        if np.asarray(layer.faces).size:
            pts = pts[np.unique(np.asarray(layer.faces).ravel())]
    elif isinstance(layer, Points):
        pts = np.asarray(layer.points)
    elif isinstance(layer, Lines):
        if not layer.polylines:
            return None
        pts = np.concatenate([np.asarray(p) for p in layer.polylines])
    elif isinstance(layer, Boxes):
        pts = np.asarray(layer.centers)
    elif isinstance(layer, Segments):
        pts = np.asarray(layer.segments).reshape(-1, 3)
    elif isinstance(layer, WireBox):
        pts = np.asarray(layer.corners)
    else:
        return None
    if pts.size == 0:
        return None
    return np.stack([pts.min(axis=0), pts.max(axis=0)])


def panel_bounds(panel: Panel) -> np.ndarray:
    """Return the ``(2, 3)`` bounds framing ``panel``."""
    if panel.bounds is not None:
        return np.asarray(panel.bounds, dtype=np.float64)
    for layer in panel.layers:
        bounds = _layer_bounds(layer)
        if bounds is not None:
            return bounds
    return np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])


def _scalar_bar_args() -> ScalarBarArgs:
    return {
        "title": "",
        "color": TEXT,
        "vertical": False,
        "width": 0.6,
        "height": 0.06,
        "position_x": 0.2,
        "position_y": 0.04,
        "title_font_size": 20,
        "label_font_size": 16,
        "fmt": "%.3g",
        "n_labels": 3,
    }


def _add_layer(plotter: pv.Plotter, layer: Layer, diagonal: float) -> None:
    if isinstance(layer, Mesh):
        surface = polydata(layer.vertices, layer.faces)
        kwargs: dict[str, Any] = {
            "show_edges": layer.show_edges,
            "edge_color": layer.edge_color,
            "line_width": layer.line_width,
            "opacity": layer.opacity,
            "smooth_shading": layer.smooth,
            "specular": 0.25,
            "split_sharp_edges": layer.smooth,
        }
        if layer.face_colors is not None:
            surface.cell_data["rgb"] = np.asarray(layer.face_colors)
            plotter.add_mesh(surface, scalars="rgb", rgb=True, **kwargs)
        elif layer.scalars is not None:
            values = np.asarray(layer.scalars)
            if layer.rgb:
                key = "rgb"
            else:
                key = layer.scalar_bar or "value"
            if values.shape[0] == surface.n_points:
                surface.point_data[key] = values
                preference = "point"
            else:
                surface.cell_data[key] = values
                preference = "cell"
            plotter.add_mesh(
                surface,
                scalars=key,
                preference=preference,
                cmap=layer.cmap,
                clim=layer.clim,
                rgb=layer.rgb,
                interpolate_before_map=True,
                show_scalar_bar=layer.scalar_bar is not None,
                scalar_bar_args=_scalar_bar_args(),
                **kwargs,
            )
        else:
            plotter.add_mesh(surface, color=layer.color, **kwargs)
    elif isinstance(layer, Lines):
        radius = diagonal * 1e-3 * layer.width
        for k, points in enumerate(layer.polylines):
            if np.asarray(points).shape[0] < 2:
                continue
            line = _polyline(points, layer.closed)
            tube = cast("pv.PolyData", line.tube(radius=radius, n_sides=10))
            if layer.scalars is not None:
                line.point_data["value"] = np.asarray(layer.scalars[k])
                tube = cast("pv.PolyData", line.tube(radius=radius, n_sides=10))
                plotter.add_mesh(tube, scalars="value", cmap=layer.cmap, show_scalar_bar=False)
            else:
                plotter.add_mesh(tube, color=layer.color, smooth_shading=True)
    elif isinstance(layer, Segments):
        segments = np.asarray(layer.segments, dtype=np.float64).reshape(-1, 2, 3)
        if segments.shape[0]:
            n = segments.shape[0]
            lines = np.hstack([np.full((n, 1), 2), np.arange(2 * n).reshape(n, 2)]).ravel()
            data = pv.PolyData(segments.reshape(-1, 3), lines=lines)
            plotter.add_mesh(
                cast("pv.PolyData", data.tube(radius=diagonal * 1e-3 * layer.width, n_sides=8)),
                color=layer.color,
                smooth_shading=True,
            )
    elif isinstance(layer, Points):
        cloud = pv.PolyData(np.asarray(layer.points, dtype=np.float64))
        kwargs = {
            "render_points_as_spheres": True,
            "point_size": layer.size,
            "opacity": layer.opacity,
        }
        if layer.scalars is not None:
            cloud.point_data["value"] = np.asarray(layer.scalars)
            plotter.add_mesh(
                cloud,
                scalars="value",
                cmap=layer.cmap,
                clim=layer.clim,
                rgb=layer.rgb,
                show_scalar_bar=layer.scalar_bar is not None,
                scalar_bar_args=_scalar_bar_args(),
                **kwargs,
            )
        else:
            plotter.add_mesh(cloud, color=layer.color, **kwargs)
    elif isinstance(layer, Arrows):
        cloud = pv.PolyData(np.asarray(layer.origins, dtype=np.float64))
        cloud.point_data["v"] = np.asarray(layer.vectors, dtype=np.float64)
        glyphs = cast(
            "pv.PolyData", cloud.glyph(orient="v", scale="v", factor=layer.scale, geom=pv.Arrow())
        )
        plotter.add_mesh(glyphs, color=layer.color, smooth_shading=True)
    elif isinstance(layer, Spheres):
        for center, radius in zip(np.asarray(layer.centers), np.asarray(layer.radii), strict=True):
            sphere = pv.Sphere(
                radius=float(radius), center=center, theta_resolution=32, phi_resolution=32
            )
            plotter.add_mesh(sphere, color=layer.color, opacity=layer.opacity, smooth_shading=True)
    elif isinstance(layer, Boxes):
        centers = np.asarray(layer.centers, dtype=np.float64)
        if centers.shape[0]:
            cloud = pv.PolyData(centers)
            if layer.scalars is not None:
                cloud.point_data["value"] = np.asarray(layer.scalars)
            cubes = cast(
                "pv.PolyData",
                cloud.glyph(
                    geom=pv.Cube(x_length=layer.size, y_length=layer.size, z_length=layer.size),
                    scale=False,
                    orient=False,
                ),
            )
            kwargs = {
                "show_edges": layer.show_edges,
                "edge_color": "#303030",
                "line_width": 0.5,
                "opacity": layer.opacity,
            }
            if layer.scalars is not None:
                plotter.add_mesh(
                    cubes, scalars="value", cmap=layer.cmap, show_scalar_bar=False, **kwargs
                )
            else:
                plotter.add_mesh(cubes, color=layer.color, **kwargs)
    elif isinstance(layer, WireBox):
        c = np.asarray(layer.corners, dtype=np.float64)
        edges = [
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        ]
        segments = np.stack([[c[a], c[b]] for a, b in edges])
        _add_layer(plotter, Segments(segments, color=layer.color, width=layer.width), diagonal)
    else:
        plane = pv.Plane(
            center=layer.center, direction=layer.normal, i_size=layer.size, j_size=layer.size
        )
        plotter.add_mesh(plane, color=layer.color, opacity=layer.opacity)


def render_panel(
    panel: Panel, camera: Camera, bounds: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    """Render one 3-D panel to an ``(h, w, 4)`` RGBA array."""
    plotter = pv.Plotter(off_screen=True, window_size=list(size), lighting="three lights")
    diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
    for layer in panel.layers:
        _add_layer(plotter, layer, diagonal)
    camera = panel.camera or camera
    center = bounds.mean(axis=0)
    direction = np.asarray(camera.direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    plotter.camera_position = [
        tuple(center + direction * diagonal * 2.0),
        tuple(center),
        tuple(camera.up),
    ]
    if camera.parallel:
        plotter.renderer.enable_parallel_projection()
    flat = bounds.T.ravel()  # xmin, xmax, ymin, ymax, zmin, zmax
    plotter.renderer.reset_camera(bounds=tuple(float(v) for v in flat))
    plotter.render()
    plotter.camera.zoom(camera.zoom)
    image = plotter.screenshot(transparent_background=True, return_img=True)
    plotter.close()
    return np.asarray(image)


def _draw_table(ax: Any, table: Table) -> None:
    ax.axis("off")
    rows = [list(r) for r in table.rows]
    widget = ax.table(
        cellText=rows[1:] if table.header else rows,
        colLabels=rows[0] if table.header else None,
        loc="center",
        cellLoc="center",
    )
    widget.auto_set_font_size(False)
    widget.set_fontsize(12)
    widget.scale(1.0, 1.6)
    for cell in widget.get_celld().values():
        cell.set_edgecolor(TEXT)
        cell.set_facecolor("none")
        cell.get_text().set_color(TEXT)


def render_figure(figure: Figure, path: Path, thumb: Path | None = None) -> None:
    """Render ``figure`` to a transparent PNG at ``path`` (plus an optional thumbnail)."""
    panels = list(figure.panels)
    ncols = figure.ncols or len(panels)
    nrows = (len(panels) + ncols - 1) // ncols
    width, height = figure.panel_size
    shared = None
    for panel in panels:
        if isinstance(panel, Panel):
            shared = panel_bounds(panel)
            break
    dpi = 100
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * width / dpi, nrows * (height + 50) / dpi),
        dpi=dpi,
        squeeze=False,
    )
    fig.patch.set_alpha(0.0)
    for ax in axes.ravel():
        ax.axis("off")
        ax.patch.set_alpha(0.0)
    for k, panel in enumerate(panels):
        ax = axes.ravel()[k]
        if isinstance(panel, Panel):
            bounds = (
                shared
                if figure.link_bounds and shared is not None and panel.bounds is None
                else panel_bounds(panel)
            )
            image = render_panel(panel, figure.camera, bounds, figure.panel_size)
            ax.imshow(image)
        elif isinstance(panel, Plot):
            ax.axis("on")
            panel.draw(ax)
            if panel.aspect:
                ax.set_aspect(panel.aspect)
            for spine in ax.spines.values():
                spine.set_color(TEXT)
            ax.tick_params(colors=TEXT)
            ax.xaxis.label.set_color(TEXT)
            ax.yaxis.label.set_color(TEXT)
        else:
            _draw_table(ax, panel)
        if panel.title:
            ax.set_title(panel.title, color=TEXT, fontsize=15)
    fig.tight_layout(pad=0.4)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", transparent=True, dpi=dpi)
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGBA")
    path.parent.mkdir(parents=True, exist_ok=True)
    _save(image, path)
    if thumb is not None:
        first = image.crop(
            (0, 0, min(image.width, image.width // ncols + 10), image.height // nrows)
        )
        first.thumbnail((360, 300))
        _save(first, thumb)


def _save(image: Image.Image, path: Path) -> None:
    """Save as lossy WebP with alpha: smooth shading survives, unlike a 256-colour PNG palette."""
    image.save(path, format="WEBP", quality=88, alpha_quality=100, method=6, exact=True)
