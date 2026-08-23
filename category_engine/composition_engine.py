"""Pure image composition helpers used only when a category template is enabled."""

from __future__ import annotations

from PIL import Image

from .category_template_manager import CategoryTemplate


def compose_with_template(
    image: Image.Image,
    size: tuple[int, int],
    background: tuple[int, int, int],
    template: CategoryTemplate,
) -> Image.Image:
    working = image.copy()
    available = (
        max(1, round(size[0] * template.scale)),
        max(1, round(size[1] * template.scale)),
    )
    working.thumbnail(available, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, background)
    center = (round(size[0] * template.position_x), round(size[1] * template.position_y))
    position = (center[0] - working.width // 2, center[1] - working.height // 2)
    canvas.paste(working, position, working if working.mode == "RGBA" else None)
    return canvas
