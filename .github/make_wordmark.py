"""Build the ordito wordmark and the GitHub social preview from the ring mark.

The ring (``docs/assets/ordito.svg``) is the initial O; "rdito" and the tagline are Outfit
(SIL OFL 1.1, https://github.com/google/fonts/tree/main/ofl/outfit) converted to outlines, so no
font is needed to view the result. Re-run after changing the ring::

    curl -Lo /tmp/Outfit.ttf "https://github.com/google/fonts/raw/main/ofl/outfit/Outfit%5Bwght%5D.ttf"
    uv run --group bench python .github/make_wordmark.py . /tmp/Outfit.ttf 600 /tmp
    uv run --group bench python -c "import cairosvg; cairosvg.svg2png(url='/tmp/social.svg', \\
        write_to='docs/assets/social-preview.png', output_width=1280)"
"""

import re
import sys
from pathlib import Path

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

repo = Path(sys.argv[1])
font = instantiateVariableFont(TTFont(sys.argv[2]), {"wght": float(sys.argv[3])})
glyphs = font.getGlyphSet()
cmap = font.getBestCmap()
upm = font["head"].unitsPerEm

ring_svg = (repo / "docs/assets/ordito.svg").read_text()
ring_paths = "".join(re.findall(r"<path [^>]*/>", ring_svg))
rx, ry, rw, rh = map(float, re.search(r'viewBox="([^"]*)"', ring_svg).group(1).split())

# The ring is the capital O: take the font's own "O" extent, overshoot included, so it sits on
# the baseline and reaches the cap height like the letter it replaces.
bp = BoundsPen(glyphs)
glyphs[cmap[ord("O")]].draw(bp)
o_xmin, o_ymin, o_xmax, o_ymax = bp.bounds
o_height = o_ymax - o_ymin
scale = o_height / rh
ring_w = rw * scale
o_lsb = o_xmin
o_rsb = glyphs[cmap[ord("O")]].width - o_xmax

# Font units, y up. Lay out: lsb, ring, rsb, then the rest of the word.
x = o_lsb
ring_x = x
x += ring_w + o_rsb
pen = SVGPathPen(glyphs)
for ch in "rdito":
    g = glyphs[cmap[ord(ch)]]
    g.draw(TransformPen(pen, (1, 0, 0, 1, x, 0)))
    x += g.width
text_d = pen.getCommands()

# Measure the outlined text for a tight box.
tb = BoundsPen(None)
x2 = o_lsb + ring_w + o_rsb
for ch in "rdito":
    g = glyphs[cmap[ord(ch)]]
    g.draw(TransformPen(tb, (1, 0, 0, 1, x2, 0)))
    x2 += g.width
t_xmin, t_ymin, t_xmax, t_ymax = tb.bounds
top = max(t_ymax, o_ymax)
bottom = min(t_ymin, o_ymin)
pad = 0.06 * upm
left = ring_x - pad
width = t_xmax + pad - left
height = top - bottom + 2 * pad


def wordmark(fill: str, note: str) -> str:
    ring_tf = (
        f"translate({ring_x - left:.2f} {top + pad - o_ymax:.2f}) "
        f"scale({scale:.5f}) translate({-rx:.2f} {-ry:.2f})"
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.1f} {height:.1f}" '
        f'role="img" aria-label="ordito">\n'
        f"  <title>ordito</title>\n"
        f"  <!-- {note} The O is the ring mark (ordito.svg), scaled to the extent of the\n"
        f"       font's own capital O; the letters are Outfit (SIL OFL 1.1) converted to\n"
        f"       outlines, so the wordmark renders the same with no font installed. -->\n"
        f'  <g transform="{ring_tf}">{ring_paths}</g>\n'
        f'  <path fill="{fill}" transform="translate({-left:.2f} {top + pad:.2f}) scale(1 -1)" '
        f'd="{text_d}"/>\n'
        f"</svg>\n"
    )


out = repo / "docs/assets"
(out / "wordmark.svg").write_text(wordmark("#15200A", "For light backgrounds."))
(out / "wordmark-dark.svg").write_text(
    wordmark("#F2F5EC", "For dark backgrounds: only the letter fill differs.")
)
print(f"viewBox {width:.0f} x {height:.0f}, aspect {width / height:.2f}")


def outline(text: str, weight: float) -> tuple[str, float]:
    """Return ``text`` as an SVG path in font units (y up) and its advance width."""
    f = instantiateVariableFont(TTFont(sys.argv[2]), {"wght": weight})
    gs, cm = f.getGlyphSet(), f.getBestCmap()
    pen, adv = SVGPathPen(gs), 0.0
    for ch in text:
        g = gs[cm[ord(ch)]]
        g.draw(TransformPen(pen, (1, 0, 0, 1, adv, 0)))
        adv += g.width
    return pen.getCommands(), adv


# GitHub social preview, 1280 x 640: the dark wordmark over the docs header green, a tagline below.
W, H = 1280, 640
mark_w = 860
k = mark_w / width
mark_h = height * k
tag_d, tag_adv = outline("GPU mesh processing on NVIDIA Warp", 400)
tag_k = 44 / upm
inner = re.search(r"<svg[^>]*>(.*)</svg>", wordmark("#F2F5EC", ""), re.S).group(1)
inner = re.sub(r"<title>.*?</title>|<!--.*?-->", "", inner, flags=re.S)
mark_y = (H - mark_h) / 2 - 50
social = (
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">'
    f'<rect width="{W}" height="{H}" fill="#15200A"/>'
    f'<g transform="translate({(W - mark_w) / 2:.2f} {mark_y:.2f}) scale({k:.5f})">{inner}</g>'
    f'<path fill="#B5C99A" transform="translate({(W - tag_adv * tag_k) / 2:.2f} '
    f'{mark_y + mark_h + 70:.2f}) scale({tag_k:.5f} {-tag_k:.5f})" d="{tag_d}"/></svg>'
)
(Path(sys.argv[4]) / "social.svg").write_text(social)
