"""Generate the vecshift logo files from one definition.

Run from the repository root:

    uv run --with fonttools --with uharfbuzz python scripts/make_logo.py

The wordmark is set in Inter Display SemiBold (SIL Open Font License) and converted to
outlines, so the SVGs look the same everywhere without the font installed. Set
INTER_DISPLAY_SEMIBOLD to the font file if it isn't at the default path.
"""

from __future__ import annotations

import os
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

FONT = Path(
    os.environ.get(
        "INTER_DISPLAY_SEMIBOLD", "/usr/share/fonts/opentype/inter/InterDisplay-SemiBold.otf"
    )
)
OUT = Path("docs/images/logo")

# The mark: points moving from the old embedding space (grey) into the new one (blue),
# on a 32 x 32 grid.
DOTS = [(6.0, 25.0, 2.6), (11.2, 20.0, 3.2), (17.6, 13.8, 4.0), (25.0, 7.0, 5.0)]

THEMES = {
    # (old grey, new blue, wordmark ink)
    "light": ("#a9a8a1", "#2a78d6", "#0b0b0b"),
    "dark": ("#8a8983", "#3987e5", "#ffffff"),
}


def _oklab(hex_color: str) -> tuple[float, float, float]:
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(int(hex_color[i : i + 2], 16) / 255) for i in (1, 3, 5))
    l_ = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m_ = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s_ = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def _hex(lab: tuple[float, float, float]) -> str:
    L, a, b = lab
    l_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = (
        4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
        -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
        -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,
    )

    def enc(c: float) -> int:
        c = min(1.0, max(0.0, c))
        c = 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
        return round(c * 255)

    return "#" + "".join(f"{enc(c):02x}" for c in rgb)


def ramp(start: str, end: str, steps: int) -> list[str]:
    """Evenly spaced colours from ``start`` to ``end``, mixed in OKLab."""
    a, b = _oklab(start), _oklab(end)
    return [
        _hex(tuple(x + (y - x) * i / (steps - 1) for x, y in zip(a, b, strict=True)))  # type: ignore[arg-type]
        for i in range(steps)
    ]


def mark_paths(old: str, new: str, dx: float = 0, dy: float = 0, scale: float = 1) -> str:
    colors = ramp(old, new, len(DOTS))
    return "".join(
        f'<circle cx="{dx + x * scale:.2f}" cy="{dy + y * scale:.2f}" r="{r * scale:.2f}" '
        f'fill="{c}"/>'
        for (x, y, r), c in zip(DOTS, colors, strict=True)
    )


def wordmark(text: str, size: float) -> tuple[list[tuple[str, int, int]], float, float, float]:
    """Outline ``text``; returns per-glyph (path, start, end) char spans, width, ascent, descent."""
    font = TTFont(FONT)
    upm = font["head"].unitsPerEm
    glyphs = font.getGlyphSet()
    blob = hb.Blob.from_file_path(str(FONT))
    hb_font = hb.Font(hb.Face(blob))
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hb_font, buf, {"kern": True, "liga": False})
    scale = size / upm
    tracking = -0.02 * size  # slightly tight, as display type usually is
    x = 0.0
    order = font.getGlyphOrder()
    out: list[tuple[str, int, int]] = []
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions, strict=True):
        pen = SVGPathPen(glyphs)
        glyphs[order[info.codepoint]].draw(
            TransformPen(pen, (scale, 0, 0, -scale, x + pos.x_offset * scale, 0))
        )
        out.append((pen.getCommands(), info.cluster, info.cluster + 1))
        x += pos.x_advance * scale + tracking
    os2 = font["OS/2"]
    return out, x - tracking, os2.sCapHeight * scale, -os2.sTypoDescender * scale


def lockup(theme: str) -> str:
    old, new, ink = THEMES[theme]
    size = 40.0
    glyphs, width, cap, _ = wordmark("vecshift", size)
    mark_size = 44.0
    gap = 12.0
    height = mark_size
    baseline = height / 2 + cap / 2  # optically centre the cap height on the mark
    split = len("vec")
    text = "".join(
        f'<path d="{d}" fill="{ink if start < split else new}"/>' for d, start, _ in glyphs
    )
    total = mark_size + gap + width
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total:.1f} {height:.1f}" '
        f'width="{total * 2:.0f}" height="{height * 2:.0f}" role="img" aria-label="vecshift">'
        f"{mark_paths(old, new, scale=mark_size / 32)}"
        f'<g transform="translate({mark_size + gap:.2f} {baseline:.2f})">{text}</g></svg>\n'
    )


def mark(theme: str) -> str:
    old, new, _ = THEMES[theme]
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="64" height="64" '
        f'role="img" aria-label="vecshift">{mark_paths(old, new)}</svg>\n'
    )


def adaptive_mark() -> str:
    """One mark that follows the light or dark theme, for favicons."""
    light, dark = ramp(*THEMES["light"][:2], len(DOTS)), ramp(*THEMES["dark"][:2], len(DOTS))
    rules = "".join(f".d{i}{{fill:{c}}}" for i, c in enumerate(light))
    dark_rules = "".join(f".d{i}{{fill:{c}}}" for i, c in enumerate(dark))
    circles = "".join(
        f'<circle class="d{i}" cx="{x}" cy="{y}" r="{r}"/>' for i, (x, y, r) in enumerate(DOTS)
    )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        f"<style>{rules}@media (prefers-color-scheme:dark){{{dark_rules}}}</style>"
        f"{circles}</svg>\n"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        (OUT / f"vecshift-{theme}.svg").write_text(lockup(theme))
        (OUT / f"vecshift-mark-{theme}.svg").write_text(mark(theme))
    (OUT / "vecshift-mark.svg").write_text(adaptive_mark())
    for theme in THEMES:
        print(theme, ramp(*THEMES[theme][:2], len(DOTS)))
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
