"""Create the redistributable default Logo overlays used by the public build.

The source artwork is deliberately simple and original: a four-tile mark plus
the text ``YOUR LOGO``. Users can replace these PNGs from PixelFlow settings.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RESOURCE_DIR = Path(__file__).resolve().parents[1] / "resources"
VARIANTS = (
    ("新logo 1440.png", (1440, 1440), (28, 34, 42, 255)),
    ("新logo 1440 2.png", (1440, 1440), (250, 252, 255, 255)),
    ("新logo 1920.png", (1440, 1920), (28, 34, 42, 255)),
    ("新logo 1920 2.png", (1440, 1920), (250, 252, 255, 255)),
)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def _draw_placeholder(path: Path, size: tuple[int, int], color: tuple[int, int, int, int]) -> None:
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    x, y = 64, 64
    tile, gap = 22, 7
    for row in range(2):
        for column in range(2):
            left = x + column * (tile + gap)
            top = y + row * (tile + gap)
            draw.rounded_rectangle((left, top, left + tile, top + tile), radius=6, fill=color)
    label_font = _font(34)
    label_x = x + tile * 2 + gap + 18
    label_y = y + 8
    draw.text((label_x, label_y), "YOUR LOGO", font=label_font, fill=color)
    canvas.save(path, "PNG")


def main() -> None:
    for name, size, color in VARIANTS:
        _draw_placeholder(RESOURCE_DIR / name, size, color)


if __name__ == "__main__":
    main()
