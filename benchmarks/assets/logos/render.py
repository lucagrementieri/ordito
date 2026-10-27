"""
Rasterize each vendored library mark (``svg/`` and ``raster/``) to a transparent PNG under ``png/``.

Run once (or after touching a source, ``raster/``, ``SIZE`` or ``MAX_ASPECT``):

    uv run python benchmarks/assets/logos/render.py

``registry.py`` and ``tools/render_docs.py`` only read the PNGs this writes, so ``cairosvg`` is
needed only to regenerate them.

Most vendored marks are SVG (a project's own icon-only mark, read from ``svg/``); a project whose
usable mark is a PNG (MeshLab, PyTorch3D) has a pre-rasterized source under ``raster/``, loaded with
Pillow. Each mark is cropped to its visible pixels and scaled to ``SIZE`` tall at its own aspect
ratio (at most ``MAX_ASPECT`` times as wide), so the docs' inline SVG charts draw every logo at one
height and a wordmark such as PyVista's keeps a readable width. The page theme decides the
background, and ``docs/stylesheets/extra.css`` recolours the one mark that needs it (Open3D's black
one on the dark theme).
"""

from __future__ import annotations

import io
from pathlib import Path

import cairosvg
from PIL import Image

# Pixels tall; the charts draw a logo at 18 px, so this stays crisp on a HiDPI screen.
SIZE = 96

# The widest a mark may be, relative to its height (PyVista's wordmark is about 3.6).
MAX_ASPECT = 4.0

HERE = Path(__file__).parent
SVG_DIR = HERE / "svg"
RASTER_DIR = HERE / "raster"
# ordito's own mark is the site logo, read where it lives rather than kept as a second copy here.
SUBJECT_SVG = HERE.parent.parent.parent / "docs" / "assets" / "logo.svg"
PNG_DIR = HERE / "png"


def _clear(source: Image.Image, size: int = SIZE) -> Image.Image:
    """Crop `source` to its visible pixels and scale it to `size` tall, keeping its aspect ratio."""
    image = source.convert("RGBA")
    box = image.getchannel("A").getbbox()
    if box is not None:
        image = image.crop(box)
    width = min(round(size * image.width / image.height), round(size * MAX_ASPECT))
    return image.resize((width, size), Image.Resampling.LANCZOS)


def render_svg(svg_path: Path, png_path: Path, size: int = SIZE) -> None:
    """Rasterize one SVG to a transparent PNG ``size`` tall."""
    # Render oversized, then crop and scale: asking cairosvg for the exact output size would
    # stretch a non-square source.
    raw = cairosvg.svg2png(url=str(svg_path), output_width=size * 4, output_height=size * 4)
    assert raw is not None  # no ``write_to``, so cairosvg returns the bytes
    _clear(Image.open(io.BytesIO(raw)), size).save(png_path)


def render_raster(image_path: Path, png_path: Path, size: int = SIZE) -> None:
    """Fit one pre-rasterized source (PNG/JPG) to a transparent PNG ``size`` tall."""
    with Image.open(image_path) as source:
        _clear(source, size).save(png_path)


def main() -> None:
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    svg_paths = sorted(SVG_DIR.glob("*.svg"))
    if "ordito" in {p.stem for p in svg_paths}:
        raise SystemExit("svg/ordito.svg would shadow docs/assets/logo.svg, ordito's only mark")
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
    render_svg(SUBJECT_SVG, PNG_DIR / "ordito.png")
    print(f"  {SUBJECT_SVG.relative_to(HERE.parent.parent.parent)} -> assets/logos/png/ordito.png")
    for image_path in raster_paths:
        png_path = PNG_DIR / f"{image_path.stem}.png"
        render_raster(image_path, png_path)
        print(f"  {image_path.relative_to(HERE)} -> {png_path.relative_to(HERE.parent.parent)}")


if __name__ == "__main__":
    main()
