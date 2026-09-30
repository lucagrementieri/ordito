"""
Rasterize each vendored library mark (``svg/`` and ``raster/``) to a fixed-size, square PNG.

Run once (or after touching a source, ``raster/``, or ``SIZE``):

    uv run python benchmarks/assets/logos/render.py

``registry.py`` and ``plot_comparison.py`` only ever read the PNGs this writes -- neither needs
``cairosvg`` at chart-render time, only this script does, and only when regenerating.

Most vendored marks are SVG (a project's own icon-only mark, read from ``svg/``), but not every
project publishes one -- MeshLab's own site links only a PNG, so ``raster/`` holds pre-rasterized
sources for those, loaded directly with Pillow rather than through cairosvg. Both paths converge on
the same `_fit_and_chip`: every source mark is either exactly square already or, for triwarp's own
lockup, wider than it is tall, and rather than distort the wide ones to fit a square, this scales
each to fit inside ``SIZE x SIZE`` at its native aspect ratio and centers it -- so every PNG this
writes is ``SIZE x SIZE`` regardless of the source's own proportions or format, which is the
property a chart placing these as fixed-size axis glyphs actually needs.

A neutral mid-gray chip is composited *behind* every mark, baked into the PNG rather than left to
the chart to add -- one mark (PyTorch3D's icon) is drawn in pure white with no dark element at
all, which is legible on that project's own dark navy site and invisible against
`plot_comparison.py`'s light-mode white chart surface with no chip behind it. A chip sized and
colored consistently for every mark, not special-cased for that one, is what keeps the chart
script's placement logic identical for every library regardless of what background each project
originally designed its mark against.
"""

from __future__ import annotations

import io
from pathlib import Path

import cairosvg
from PIL import Image, ImageDraw

# 96px source, ~2x an anticipated ~48px on-chart display size, so the glyph stays crisp on a
# retina/HiDPI figure export (`dpi=200`+, which `plot_comparison.py` uses throughout).
SIZE = 96

# A light, low-chroma chip -- legible against both this project's light (#ffffff) and dark
# (#14161b) chart surfaces, and neutral enough not to compete with any of the marks' own colors.
CHIP_COLOR = (216, 216, 216, 235)
CHIP_MARGIN = 4  # px of chip visible past the mark's own bounding box, before its rounding

HERE = Path(__file__).parent
SVG_DIR = HERE / "svg"
RASTER_DIR = HERE / "raster"
PNG_DIR = HERE / "png"


def _fit_and_chip(source: Image.Image, size: int = SIZE) -> Image.Image:
    """Scale `source` (any size/aspect) to fit inside `size x size`, centered on a gray chip."""
    fitted = source.convert("RGBA")
    fitted.thumbnail((size, size), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    chip_box = (CHIP_MARGIN, CHIP_MARGIN, size - CHIP_MARGIN, size - CHIP_MARGIN)
    ImageDraw.Draw(canvas).rounded_rectangle(chip_box, radius=size * 0.18, fill=CHIP_COLOR)
    offset = ((size - fitted.width) // 2, (size - fitted.height) // 2)
    canvas.paste(fitted, offset, fitted)
    return canvas


def render_svg(svg_path: Path, png_path: Path, size: int = SIZE) -> None:
    """Rasterize one SVG to a `size x size` PNG, on a chip, centered and fitted."""
    # Render oversized on the long axis, then fit -- asking cairosvg for the exact output size
    # directly would stretch a non-square source instead of letter/pillar-boxing it.
    raw = cairosvg.svg2png(url=str(svg_path), output_width=size * 4, output_height=size * 4)
    assert raw is not None  # no ``write_to``, so cairosvg returns the bytes
    _fit_and_chip(Image.open(io.BytesIO(raw)), size).save(png_path)


def render_raster(image_path: Path, png_path: Path, size: int = SIZE) -> None:
    """Fit one pre-rasterized source (PNG/JPG) to a `size x size` PNG, on a chip."""
    with Image.open(image_path) as source:
        _fit_and_chip(source, size).save(png_path)


def main() -> None:
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    svg_paths = sorted(SVG_DIR.glob("*.svg"))
    raster_paths = sorted(p for ext in ("*.png", "*.jpg", "*.jpeg") for p in RASTER_DIR.glob(ext))
    if not svg_paths and not raster_paths:
        raise SystemExit(f"no sources found under {SVG_DIR} or {RASTER_DIR}")
    collisions = {p.stem for p in svg_paths} & {p.stem for p in raster_paths}
    if collisions:
        raise SystemExit(f"same stem in both svg/ and raster/, ambiguous: {sorted(collisions)}")

    for svg_path in svg_paths:
        png_path = PNG_DIR / f"{svg_path.stem}.png"
        render_svg(svg_path, png_path)
        print(f"  {svg_path.relative_to(HERE)} -> {png_path.relative_to(HERE.parent.parent)}")
    for image_path in raster_paths:
        png_path = PNG_DIR / f"{image_path.stem}.png"
        render_raster(image_path, png_path)
        print(f"  {image_path.relative_to(HERE)} -> {png_path.relative_to(HERE.parent.parent)}")


if __name__ == "__main__":
    main()
