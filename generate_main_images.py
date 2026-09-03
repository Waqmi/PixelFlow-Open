from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from statistics import median
from threading import Lock
from typing import Callable

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageStat

from output_validator import ValidationIssue, validate_output
from local_model_assistant import (
    LocalModelAssistant,
    credible_model_composition,
    find_compatible_onnx_model,
)

from category_engine import (
    AICategoryProvider,
    CategoryTemplate,
    CategoryTemplateManager,
    RuleCategoryClassifier,
    RuleSourceTypeClassifier,
    AISourceTypeProvider,
    SourceTypeDecision,
    compose_with_template,
)
from ai_image_assistant import OpenAIVisionAssistant


if getattr(sys, "frozen", False):
    executable_dir = Path(sys.executable).resolve().parent
    ROOT = (
        executable_dir.parent / "Resources"
        if sys.platform == "darwin"
        else Path(getattr(sys, "_MEIPASS", executable_dir))
    )
else:
    ROOT = Path(__file__).resolve().parent
RESOURCE_ROOT = ROOT / "resources"
PRODUCT_ROOT = ROOT / "产品图"
OUTPUT_ROOT = ROOT / "成品主图"
TRANSPARENT_OUTPUT_ROOT = OUTPUT_ROOT / "透明产品图"
LOGO_PATH = RESOURCE_ROOT / "新logo 1440.png"
WHITE_LOGO_PATH = RESOURCE_ROOT / "新logo 1440 2.png"
TALL_LOGO_PATH = RESOURCE_ROOT / "新logo 1920.png"
TALL_WHITE_LOGO_PATH = RESOURCE_ROOT / "新logo 1920 2.png"
BORDER_PATH = RESOURCE_ROOT / "唯品会边框.png"

SIZES = {
    "1440x1440": ((1440, 1440), (255, 255, 255)),
    "1440x1920": ((1440, 1920), (239, 239, 239)),
    "1125x1500": ((1125, 1500), (239, 239, 239)),
}

# Bump whenever rendering logic changes; this invalidates old manifests.
ENGINE_VERSION = "stable-24"

MODEL_GROUP_NAMES = {"模特图", "模特细节图", "模特主图", "model", "lookbook"}

TRANSPARENT_SIZES = {
    "800x800": (800, 800),
    "1440x1440": (1440, 1440),
    "1440x1920": (1440, 1920),
    "1440x2160": (1440, 2160),
}

# New transparent-only sizes reuse the closest existing category composition.
TRANSPARENT_TEMPLATE_SIZES = {
    "800x800": "1440x1440",
    "1440x1440": "1440x1440",
    "1440x1920": "1440x1920",
    "1440x2160": "1440x1920",
}

ProgressCallback = Callable[[int, int, str], None]

DEFAULT_RENDER_RULES = {
    "background_extension": {
        "enabled_for_opaque_main_product": True,
        "edge_sample_ratio": 0.04,
        "blend_ratio": 0.03,
        "use_corner_gradient": True,
        "protect_light_products": True,
    }
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SHORT_NAME_MAX_LENGTH = 18


def _material_dirs(product_root: Path) -> tuple[Path, ...]:
    """Return every folder that directly contains images.

    Besides the original ``root/color/image`` layout, material roots may have
    sibling groups such as ``root/model`` and ``root/product/color/image``.
    Scanning only the first child level silently dropped the nested product
    folders, so walk directories until each image-containing folder is found.
    """
    material_dirs: list[Path] = []

    def collect(directory: Path) -> None:
        try:
            entries = tuple(sorted(directory.iterdir()))
        except OSError:
            return
        if any(
            entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS
            for entry in entries
        ):
            material_dirs.append(directory)
        for entry in entries:
            if entry.is_dir() and not entry.is_symlink() and not entry.name.startswith("."):
                collect(entry)

    collect(product_root)
    return tuple(material_dirs)


def _auto_worker_count(source_count: int) -> int:
    """Choose a conservative cross-platform render concurrency automatically."""
    if source_count <= 1:
        return max(1, source_count)
    logical_cpus = os.cpu_count() or 2
    if logical_cpus <= 2:
        return 1
    if logical_cpus <= 4:
        return min(2, source_count)
    return min(3, source_count)


@dataclass
class SourceAnalysis:
    """Per-source measurements reused across all output sizes."""

    source: Path
    image: Image.Image
    has_transparency: bool
    edge_color: tuple[int, int, int] | None
    foreground_bbox: tuple[int, int, int, int] | None
    touches_edge: bool
    light_background: bool
    preserve_light_product_pixels: bool


def load_render_rules(resource_root: Path) -> dict[str, object]:
    """Load optional rendering parameters without making resources mandatory."""
    rules = json.loads(json.dumps(DEFAULT_RENDER_RULES))
    rules_path = resource_root / "templates" / "render_rules.json"
    if not rules_path.exists():
        return rules
    payload = json.loads(rules_path.read_text(encoding="utf-8"))
    extension = payload.get("background_extension")
    if isinstance(extension, dict):
        rules["background_extension"].update(extension)
    return rules


def adapt_template_for_source(
    image: Image.Image, template: CategoryTemplate
) -> CategoryTemplate:
    """Give only unusually tall shoe views a little extra breathing room."""
    if template.category == "shoes" and image.width / max(1, image.height) < 0.75:
        return replace(template, scale=round(template.scale * 0.86, 3))
    return template


def normalize_light_background(image: Image.Image, target_color: tuple[int, int, int]) -> Image.Image:
    """Replace edge-connected light background pixels without touching the product."""
    rgb = image.convert("RGB")
    scale = min(1.0, 512 / max(rgb.width, rgb.height))
    preview_size = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
    preview = rgb.resize(preview_size, Image.Resampling.BILINEAR)
    preview_pixels = list(preview.getdata())
    candidate = bytearray(
        1 if max(pixel) - min(pixel) <= 24 and sum(pixel) / 3 >= 210 else 0
        for pixel in preview_pixels
    )
    visited = bytearray(len(candidate))
    width, height = preview_size
    pending = deque()
    for x in range(width):
        if candidate[x]:
            pending.append((0, x))
        if candidate[(height - 1) * width + x]:
            pending.append((height - 1, x))
    for y in range(height):
        if candidate[y * width]:
            pending.append((y, 0))
        if candidate[y * width + width - 1]:
            pending.append((y, width - 1))
    while pending:
        y, x = pending.popleft()
        index = y * width + x
        if visited[index] or not candidate[index]:
            continue
        visited[index] = 1
        if y > 0:
            pending.append((y - 1, x))
        if y + 1 < height:
            pending.append((y + 1, x))
        if x > 0:
            pending.append((y, x - 1))
        if x + 1 < width:
            pending.append((y, x + 1))
    mask = Image.frombytes("L", preview_size, bytes(255 if value else 0 for value in visited)).resize(
        rgb.size, Image.Resampling.NEAREST
    )
    target = Image.new("RGB", rgb.size, target_color)
    return Image.composite(target, rgb, mask)


def resize_contain(
    image: Image.Image,
    size: tuple[int, int],
    background: tuple[int, int, int],
    padding_ratio: float = 0.0,
    template: CategoryTemplate | None = None,
) -> Image.Image:
    if template is not None:
        return compose_with_template(
            image, size, background, adapt_template_for_source(image, template)
        )
    available_size = (
        max(1, round(size[0] * (1 - padding_ratio * 2))),
        max(1, round(size[1] * (1 - padding_ratio * 2))),
    )
    image.thumbnail(available_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, background)
    position = ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
    canvas.paste(image, position, image if image.mode == "RGBA" else None)
    return canvas


def resize_transparent_with_template(
    image: Image.Image,
    size: tuple[int, int],
    template: CategoryTemplate,
) -> Image.Image:
    working = image.copy()
    alpha_box = working.getchannel("A").getbbox()
    if alpha_box:
        working = crop_with_margin(working, alpha_box)
    template = adapt_template_for_source(working, template)
    available = (
        max(1, round(size[0] * template.scale)),
        max(1, round(size[1] * template.scale)),
    )
    working.thumbnail(available, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    center = (round(size[0] * template.position_x), round(size[1] * template.position_y))
    position = (center[0] - working.width // 2, center[1] - working.height // 2)
    canvas.alpha_composite(working, position)
    return canvas


def resize_cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    scale = max(size[0] / image.width, size[1] / image.height)
    resized = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - size[0]) // 2
    top = (resized.height - size[1]) // 2
    return resized.crop((left, top, left + size[0], top + size[1]))


def model_background_extension_safe(
    image: Image.Image,
    size: tuple[int, int],
    person_bboxes: list[tuple[float, float, float, float]] | None = None,
) -> bool:
    """Allow mirror extension only when the required side edges are clean.

    This intentionally checks the two side strips row by row instead of using
    the whole-image foreground box. The inspected strip is as wide as the
    extension that would actually be needed; this catches arms or garments
    near the edge even when they are symmetrically placed.
    """
    source = image.convert("RGB")
    if not person_bboxes:
        # Without a detector box, do not risk mirroring an unknown object.
        return False
    leftmost = min(float(box[0]) for box in person_bboxes)
    rightmost = max(float(box[2]) for box in person_bboxes)
    safety_margin = source.width * 0.01
    if leftmost <= safety_margin or source.width - rightmost <= safety_margin:
        return False

    # Reject visibly textured edge strips. A clean wall, floor, or gentle
    # lighting gradient can pass; a prop, foliage, text, or hard background
    # detail should stay on the cover-crop path.
    preview_width = min(256, source.width)
    preview_height = max(1, round(source.height * preview_width / source.width))
    preview = source.resize((preview_width, preview_height), Image.Resampling.BILINEAR)
    strip = max(2, round(preview_width * 0.01))
    noisy_rows = 0
    for y in range(preview_height):
        for start in (0, preview_width - strip):
            pixels = [preview.getpixel((x, y)) for x in range(start, start + strip)]
            channel_range = max(
                max(pixel[channel] for pixel in pixels)
                - min(pixel[channel] for pixel in pixels)
                for channel in range(3)
            )
            if channel_range > 25:
                noisy_rows += 1
    return noisy_rows / max(1, preview_height * 2) <= 0.03


def mirror_side_fill(
    image: Image.Image,
    size: tuple[int, int],
    left: int,
    top: int,
    edge_source_widths: tuple[int, int] | None = None,
) -> Image.Image:
    """Fill uncovered canvas areas from the source image's edge pixels."""
    canvas = Image.new("RGB", size)
    right = left + image.width
    bottom = top + image.height
    if left >= 0:
        extension_width = min(left, image.width)
        source_width = min(
            extension_width,
            edge_source_widths[0] if edge_source_widths else extension_width,
        )
        if extension_width and source_width:
            left_extension = image.crop((0, 0, source_width, image.height))
            left_extension = ImageOps.mirror(left_extension).resize(
                (extension_width, image.height), Image.Resampling.BILINEAR
            )
            canvas.paste(left_extension, (0, top))
    if right <= size[0]:
        extension_width = min(size[0] - right, image.width)
        source_width = min(
            extension_width,
            edge_source_widths[1] if edge_source_widths else extension_width,
        )
        if extension_width and source_width:
            right_extension = image.crop(
                (image.width - source_width, 0, image.width, image.height)
            )
            right_extension = ImageOps.mirror(right_extension).resize(
                (extension_width, image.height), Image.Resampling.BILINEAR
            )
            canvas.paste(right_extension, (right, top))
    if top >= 0:
        top_extension = image.crop((0, 0, image.width, min(top, image.height)))
        if top_extension.height:
            canvas.paste(ImageOps.flip(top_extension), (left, top - top_extension.height))
    if bottom <= size[1]:
        extension_height = min(size[1] - bottom, image.height)
        if extension_height:
            bottom_extension = image.crop((0, image.height - extension_height, image.width, image.height))
            canvas.paste(ImageOps.flip(bottom_extension), (left, bottom))
    paste_box = (max(0, left), max(0, top), min(size[0], right), min(size[1], bottom))
    source_box = (
        max(0, -left),
        max(0, -top),
        max(0, -left) + paste_box[2] - paste_box[0],
        max(0, -top) + paste_box[3] - paste_box[1],
    )
    if paste_box[2] > paste_box[0] and paste_box[3] > paste_box[1]:
        canvas.paste(image.crop(source_box), (paste_box[0], paste_box[1]))
    return canvas


def resize_model_preserve(
    image: Image.Image,
    size: tuple[int, int],
    scale_adjustment: float = 0.0,
    position_x: float = 0.5,
    position_y: float = 0.5,
    allow_mirror_extension: bool = False,
    person_bboxes: list[tuple[float, float, float, float]] | None = None,
    target_crop_height_ratio: float | None = None,
    protected_crop_bbox: tuple[float, float, float, float] | None = None,
) -> Image.Image:
    """Compose a model photo without borders, extending only clean side edges."""
    source = image.convert("RGB")
    can_extend_background = (
        size[0] == size[1]
        and model_background_extension_safe(source, size, person_bboxes)
    )
    # A square crop of a 2:3 model source otherwise enlarges the person just
    # to fill the width. Fit by the limiting dimension when clean side edges
    # are available, then mirror those background-only edge strips.
    direct_target_scale = size[0] == size[1] and target_crop_height_ratio
    if direct_target_scale:
        crop_ratio = max(0.20, min(1.0, float(target_crop_height_ratio)))
        # Full-body shirt shots use a deliberate square crop from the head
        # toward the hip. Calculate that crop directly instead of applying a
        # cover-based adjustment to a contained photo, which made the person
        # visibly too small in samples such as 03, 04, and 18.
        scale = size[1] / (source.height * crop_ratio)
        scale_adjustment = 0.0
    else:
        scale = (
            min(size[0] / source.width, size[1] / source.height)
            if can_extend_background
            else max(size[0] / source.width, size[1] / source.height)
        )
    # Full-body sources often need a deliberate 2× crop to turn a clothing
    # photo into a usable main image.  The composition assistant already
    # derives this from the detected body area, so do not silently cap it at
    # 25% here (that was why full-body images still looked almost unchanged).
    scale *= 1 + max(-10.0, min(200.0, scale_adjustment)) / 100
    if protected_crop_bbox:
        # A shoe composition may only zoom as far as the complete protected
        # shoe region can still fit inside the target canvas.
        protected_left, protected_top, protected_right, protected_bottom = (
            max(0.0, min(float(value), float(limit)))
            for value, limit in zip(
                protected_crop_bbox,
                (source.width, source.height, source.width, source.height),
            )
        )
        protected_width = max(1.0, protected_right - protected_left)
        protected_height = max(1.0, protected_bottom - protected_top)
        maximum_safe_scale = min(
            size[0] / protected_width,
            size[1] / protected_height,
        )
        scale = min(scale, maximum_safe_scale)
    fitted = source.resize(
        (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
        Image.Resampling.LANCZOS,
    )
    position_x = max(0.0, min(1.0, position_x))
    position_y = max(0.0, min(1.0, position_y))
    if fitted.width >= size[0] and fitted.height >= size[1]:
        left = round((fitted.width - size[0]) * position_x)
        top = round((fitted.height - size[1]) * position_y)
        if protected_crop_bbox:
            # Clamp the crop origin so every edge of the protected region is
            # inside the output, even when the pose shifted the focus.
            protected_left, protected_top, protected_right, protected_bottom = (
                float(value) for value in protected_crop_bbox
            )
            left_min = max(0, round(protected_right * scale - size[0]))
            left_max = min(fitted.width - size[0], round(protected_left * scale))
            top_min = max(0, round(protected_bottom * scale - size[1]))
            top_max = min(fitted.height - size[1], round(protected_top * scale))
            left = max(left_min, min(left_max, left))
            top = max(top_min, min(top_max, top))
        return fitted.crop((left, top, left + size[0], top + size[1]))
    if can_extend_background:
        if person_bboxes:
            subject_center_x = (
                min(float(box[0]) for box in person_bboxes)
                + max(float(box[2]) for box in person_bboxes)
            ) / 2.0
            # The detector's position_x was calculated for cover cropping;
            # reuse it here would shift the contained photo toward one side.
            left = round(size[0] * 0.5 - subject_center_x * scale)
        else:
            left = round((size[0] - fitted.width) * position_x)
        top = round((size[1] - fitted.height) * position_y)
        fit_scale = max(0.0001, scale)
        edge_strip_limit = max(1, round(source.width * 0.01))
        edge_widths = (
            max(1, round(min(left / fit_scale, edge_strip_limit, source.width))),
            max(1, round(min(
                (size[0] - left - fitted.width) / fit_scale,
                edge_strip_limit,
                source.width,
            ))),
        )
        return mirror_side_fill(fitted, size, left, top, edge_widths)
    # Rounding can leave a one-pixel short side on unusual image ratios.
    return resize_cover(source, size)


def crop_with_margin(image: Image.Image, bbox: tuple[int, int, int, int], margin_ratio: float = 0.01) -> Image.Image:
    """Trim source-only whitespace while retaining a small edge safety margin."""
    left, top, right, bottom = bbox
    margin_x = round((right - left) * margin_ratio)
    margin_y = round((bottom - top) * margin_ratio)
    return image.crop(
        (
            max(0, left - margin_x),
            max(0, top - margin_y),
            min(image.width, right + margin_x),
            min(image.height, bottom + margin_y),
        )
    )


def foreground_bbox(
    image: Image.Image,
    background: tuple[int, int, int],
    tolerance: int = 32,
) -> tuple[int, int, int, int] | None:
    """Find the non-background bounds of a light-background product photo."""
    rgb = image.convert("RGB")
    scale = min(1.0, 512 / max(rgb.width, rgb.height))
    preview_size = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
    preview = rgb.resize(preview_size, Image.Resampling.BILINEAR)
    width, height = preview_size
    pixels = list(preview.getdata())
    background_candidate = bytearray(
        max(abs(channel - base) for channel, base in zip(pixel, background)) <= tolerance
        for pixel in pixels
    )
    visited = bytearray(len(background_candidate))
    pending = deque()
    for x in range(width):
        pending.append((0, x))
        pending.append((height - 1, x))
    for y in range(1, height - 1):
        pending.append((y, 0))
        pending.append((y, width - 1))
    while pending:
        y, x = pending.popleft()
        index = y * width + x
        if visited[index] or not background_candidate[index]:
            continue
        visited[index] = 1
        if y > 0:
            pending.append((y - 1, x))
        if y + 1 < height:
            pending.append((y + 1, x))
        if x > 0:
            pending.append((y, x - 1))
        if x + 1 < width:
            pending.append((y, x + 1))
    foreground = [index for index, value in enumerate(visited) if not value]
    if not foreground:
        return None
    left = min(index % width for index in foreground)
    top = min(index // width for index in foreground)
    right = max(index % width for index in foreground) + 1
    bottom = max(index // width for index in foreground) + 1
    return (
        round(left / width * rgb.width),
        round(top / height * rgb.height),
        round(right / width * rgb.width),
        round(bottom / height * rgb.height),
    )


def light_background_to_alpha(
    image: Image.Image,
    background: tuple[int, int, int],
    low_threshold: int = 8,
    high_threshold: int = 48,
    edge_feather_ratio: float = 0.08,
) -> Image.Image:
    """Turn the light background transparent while retaining soft shadows."""
    working = image.convert("RGBA")
    scale = min(1.0, 512 / max(working.width, working.height))
    preview_size = (
        max(1, round(working.width * scale)),
        max(1, round(working.height * scale)),
    )
    width, height = preview_size
    alpha = bytearray(width * height)
    pixels = list(working.convert("RGB").resize(preview_size, Image.Resampling.BILINEAR).getdata())
    for y in range(height):
        for x in range(width):
            pixel = pixels[y * width + x]
            difference = max(
                abs(channel - base) for channel, base in zip(pixel, background)
            )
            if difference <= low_threshold:
                value = 0
            elif difference >= high_threshold:
                value = 255
            else:
                value = round(
                    (difference - low_threshold)
                    / max(1, high_threshold - low_threshold)
                    * 255
                )
            if difference < high_threshold:
                edge_distance = min(
                    x / max(1, width * edge_feather_ratio),
                    y / max(1, height * edge_feather_ratio),
                    (width - 1 - x) / max(1, width * edge_feather_ratio),
                    (height - 1 - y) / max(1, height * edge_feather_ratio),
                )
                value = round(value * min(1.0, edge_distance))
            alpha[y * width + x] = value
    mask = Image.frombytes("L", preview_size, bytes(alpha)).resize(
        working.size, Image.Resampling.BICUBIC
    )
    working.putalpha(mask)
    return working


def sample_edge_color(image: Image.Image) -> tuple[int, int, int]:
    image = image.convert("RGB")
    step = max(1, max(image.width, image.height) // 256)
    edge_width = max(1, min(image.width, image.height) // 100)
    top = min(image.height - 1, edge_width // 2)
    bottom = max(0, image.height - edge_width // 2 - 1)
    left = min(image.width - 1, edge_width // 2)
    right = max(0, image.width - edge_width // 2 - 1)
    samples = []
    for x in range(0, image.width, step):
        samples.append(image.getpixel((x, top)))
        samples.append(image.getpixel((x, bottom)))
    for y in range(0, image.height, step):
        samples.append(image.getpixel((left, y)))
        samples.append(image.getpixel((right, y)))
    return tuple(int(median(channel)) for channel in zip(*samples))


def foreground_touches_edge(image: Image.Image, background: tuple[int, int, int]) -> bool:
    """Detect detail photos whose product continues into the source edge."""
    preview = image.convert("RGB")
    scale = min(1.0, 512 / max(preview.width, preview.height))
    preview = preview.resize(
        (max(1, round(preview.width * scale)), max(1, round(preview.height * scale))),
        Image.Resampling.BILINEAR,
    )
    width, height = preview.size
    tolerance = 32
    edge_pixels = []
    for x in range(width):
        edge_pixels.append(preview.getpixel((x, 0)))
        edge_pixels.append(preview.getpixel((x, height - 1)))
    for y in range(1, height - 1):
        edge_pixels.append(preview.getpixel((0, y)))
        edge_pixels.append(preview.getpixel((width - 1, y)))
    foreground_count = sum(
        max(abs(channel - base) for channel, base in zip(pixel, background)) > tolerance
        for pixel in edge_pixels
    )
    return foreground_count / max(1, len(edge_pixels)) >= 0.03


def _light_product_pixels_present(
    image: Image.Image, bbox: tuple[int, int, int, int] | None
) -> bool:
    """Flag light products so future processing never treats their pixels as background."""
    if bbox is None:
        return False
    preview = image.convert("RGB").crop(bbox)
    preview.thumbnail((256, 256), Image.Resampling.BILINEAR)
    pixels = list(preview.getdata())
    if not pixels:
        return False
    light_neutral = sum(
        sum(pixel) / 3 >= 210 and max(pixel) - min(pixel) <= 28
        for pixel in pixels
    )
    return light_neutral / len(pixels) >= 0.12


def analyze_source(source: Path) -> SourceAnalysis:
    """Read each input once and keep only non-destructive measurements."""
    with Image.open(source) as opened:
        opened.load()
        image = opened.copy()
    transparent = has_transparency(image)
    if transparent:
        alpha_box = image.convert("RGBA").getchannel("A").getbbox()
        return SourceAnalysis(
            source=source,
            image=image,
            has_transparency=True,
            edge_color=None,
            foreground_bbox=alpha_box,
            touches_edge=False,
            light_background=False,
            preserve_light_product_pixels=False,
        )
    rgb = image.convert("RGB")
    edge_color = sample_edge_color(rgb)
    light_background = max(edge_color) - min(edge_color) <= 20 and sum(edge_color) / 3 >= 220
    product_box = foreground_bbox(rgb, edge_color) if light_background else None
    touches_edge = (
        foreground_touches_edge(rgb, edge_color) if light_background else False
    )
    return SourceAnalysis(
        source=source,
        image=rgb,
        has_transparency=False,
        edge_color=edge_color,
        foreground_bbox=product_box,
        touches_edge=touches_edge,
        light_background=light_background,
        preserve_light_product_pixels=_light_product_pixels_present(rgb, product_box),
    )


def _corner_background_colors(
    image: Image.Image,
    product_box: tuple[int, int, int, int] | None,
    sample_ratio: float,
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    """Sample four robust corner colours while excluding the measured product area."""
    rgb = image.convert("RGB")
    sample_width = max(2, round(rgb.width * sample_ratio))
    sample_height = max(2, round(rgb.height * sample_ratio))
    excluded = None
    if product_box is not None:
        margin_x = round((product_box[2] - product_box[0]) * 0.02)
        margin_y = round((product_box[3] - product_box[1]) * 0.02)
        excluded = (
            max(0, product_box[0] - margin_x),
            max(0, product_box[1] - margin_y),
            min(rgb.width, product_box[2] + margin_x),
            min(rgb.height, product_box[3] + margin_y),
        )

    def sample(left: int, top: int) -> tuple[int, int, int]:
        values = []
        step = max(1, min(sample_width, sample_height) // 12)
        for y in range(top, min(rgb.height, top + sample_height), step):
            for x in range(left, min(rgb.width, left + sample_width), step):
                if excluded and excluded[0] <= x < excluded[2] and excluded[1] <= y < excluded[3]:
                    continue
                values.append(rgb.getpixel((x, y)))
        if not values:
            return sample_edge_color(rgb)
        return tuple(int(median(channel)) for channel in zip(*values))

    return (
        sample(0, 0),
        sample(max(0, rgb.width - sample_width), 0),
        sample(0, max(0, rgb.height - sample_height)),
        sample(max(0, rgb.width - sample_width), max(0, rgb.height - sample_height)),
    )


def _model_background_colors(
    image: Image.Image,
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    """Sample neutral light background colours without picking up garments."""
    rgb = image.convert("RGB")
    preview = rgb.resize((min(256, rgb.width), min(256, rgb.height)), Image.Resampling.BILINEAR)
    pixels = list(preview.getdata())

    def is_background(pixel: tuple[int, int, int]) -> bool:
        return max(pixel) - min(pixel) <= 30 and sum(pixel) / 3 >= 150

    neutral_pixels = [pixel for pixel in pixels if is_background(pixel)]
    if not neutral_pixels:
        return _corner_background_colors(rgb, None, 0.06)
    fallback = tuple(int(median(channel)) for channel in zip(*neutral_pixels))
    sample_width = max(2, round(rgb.width * 0.06))
    sample_height = max(2, round(rgb.height * 0.06))

    def sample(left: int, top: int) -> tuple[int, int, int]:
        values = []
        step = max(1, min(sample_width, sample_height) // 12)
        for y in range(top, min(rgb.height, top + sample_height), step):
            for x in range(left, min(rgb.width, left + sample_width), step):
                pixel = rgb.getpixel((x, y))
                if is_background(pixel):
                    values.append(pixel)
        return tuple(int(median(channel)) for channel in zip(*values)) if values else fallback

    return (
        sample(0, 0),
        sample(max(0, rgb.width - sample_width), 0),
        sample(0, max(0, rgb.height - sample_height)),
        sample(max(0, rgb.width - sample_width), max(0, rgb.height - sample_height)),
    )


def _corner_gradient(
    size: tuple[int, int],
    colors: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
) -> Image.Image:
    gradient = Image.new("RGB", (2, 2))
    gradient.putdata((colors[0], colors[1], colors[2], colors[3]))
    return gradient.resize(size, Image.Resampling.BILINEAR)


def _outer_background_blend_mask(
    size: tuple[int, int],
    product_box: tuple[int, int, int, int] | None,
    blend_ratio: float,
) -> Image.Image:
    """Feather only a photo's outer background edge; the product box stays opaque."""
    edge = max(1, round(min(size) * blend_ratio))
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((edge, edge, max(edge, size[0] - edge - 1), max(edge, size[1] - edge - 1)), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(1, edge // 2)))
    if product_box is not None:
        draw = ImageDraw.Draw(mask)
        draw.rectangle(product_box, fill=255)
    return mask


def _scaled_opaque_canvas(
    analysis: SourceAnalysis,
    size: tuple[int, int],
    category_template: CategoryTemplate | None,
    scale_adjustment: float,
    rules: dict[str, object],
    portrait_padding: float,
) -> Image.Image:
    """Scale the complete opaque photo and extend only its sampled background."""
    assert analysis.edge_color is not None
    image = analysis.image.convert("RGB")
    product_box = analysis.foreground_bbox
    if product_box is None:
        return resize_contain(image.copy(), size, analysis.edge_color, padding_ratio=portrait_padding)

    extension = rules["background_extension"]
    assert isinstance(extension, dict)
    base_scale = category_template.scale if category_template is not None else 1 - portrait_padding * 2
    target_scale = base_scale * (1 + scale_adjustment / 100)
    box_width = max(1, product_box[2] - product_box[0])
    box_height = max(1, product_box[3] - product_box[1])
    photo_scale = min(size[0] * target_scale / box_width, size[1] * target_scale / box_height)
    scaled_size = (
        max(1, round(image.width * photo_scale)),
        max(1, round(image.height * photo_scale)),
    )
    scaled = image.resize(scaled_size, Image.Resampling.LANCZOS)
    scaled_box = tuple(round(value * photo_scale) for value in product_box)
    position_x = category_template.position_x if category_template is not None else 0.5
    position_y = category_template.position_y if category_template is not None else 0.5
    product_center_x = (scaled_box[0] + scaled_box[2]) / 2
    product_center_y = (scaled_box[1] + scaled_box[3]) / 2
    position = (
        round(size[0] * position_x - product_center_x),
        round(size[1] * position_y - product_center_y),
    )
    colors = _corner_background_colors(
        image,
        product_box,
        float(extension.get("edge_sample_ratio", 0.04)),
    )
    canvas = _corner_gradient(size, colors) if extension.get("use_corner_gradient", True) else Image.new("RGB", size, analysis.edge_color)
    blend_mask = _outer_background_blend_mask(
        scaled_size,
        scaled_box if extension.get("protect_light_products", True) else None,
        float(extension.get("blend_ratio", 0.03)),
    )
    canvas.paste(scaled, position, blend_mask)
    return canvas


def build_canvas(
    source: Path,
    size: tuple[int, int],
    background: tuple[int, int, int],
    category_template: CategoryTemplate | None = None,
    source_type: str | None = None,
    source_scale_adjustment: float = 0.0,
    analysis: SourceAnalysis | None = None,
    render_rules: dict[str, object] | None = None,
    model_composition: dict[str, object] | None = None,
) -> Image.Image:
    analysis = analysis or analyze_source(source)
    image = analysis.image
    render_rules = render_rules or DEFAULT_RENDER_RULES
    if source_type == "model":
        model_composition = model_composition or {}
        by_size = model_composition.get("by_size")
        size_key = f"{size[0]}x{size[1]}"
        if isinstance(by_size, dict) and isinstance(by_size.get(size_key), dict):
            model_composition = {**model_composition, **by_size[size_key]}
        if model_composition.get("composition_mode") != "zoom_focus":
            model_composition = {**model_composition, "scale_adjustment_percent": 0.0}
        raw_person_bboxes = model_composition.get("person_bboxes")
        person_bboxes = (
            [tuple(float(value) for value in box[:4]) for box in raw_person_bboxes]
            if isinstance(raw_person_bboxes, list)
            and all(isinstance(box, (list, tuple)) and len(box) >= 4 for box in raw_person_bboxes)
            else None
        )
        return resize_model_preserve(
            image,
            size,
            float(model_composition.get("scale_adjustment_percent", 0.0)),
            float(model_composition.get("position_x", 0.5)),
            float(model_composition.get("position_y", 0.5)),
            bool(model_composition.get("allow_mirror_extension", False)),
            person_bboxes,
            (
                float(model_composition.get("target_crop_height_ratio"))
                if model_composition.get("target_crop_height_ratio") is not None
                else None
            ),
            (
                tuple(float(value) for value in model_composition["protected_crop_bbox"][:4])
                if isinstance(model_composition.get("protected_crop_bbox"), (list, tuple))
                and len(model_composition["protected_crop_bbox"]) >= 4
                else None
            ),
        )
    if source_type in {"detail", "model_detail"}:
        # Detail material is intentionally edge-to-edge and bypasses product templates.
        return resize_cover(image.convert("RGB"), size)
    is_portrait = size[1] > size[0]
    # 1125x1500 is used as a tighter marketplace preview. Its product scale
    # needs more breathing room than the 1440x1920 portrait output.
    portrait_padding = 0.12 if size == (1125, 1500) else 0.05
    if analysis.has_transparency:
        product = image.convert("RGBA")
        alpha_box = product.getchannel("A").getbbox()
        if alpha_box and (is_portrait or category_template is not None):
            product = crop_with_margin(product, alpha_box)
        if category_template is not None:
            category_template = replace(
                category_template,
                scale=category_template.scale * (1 + source_scale_adjustment / 100),
            )
        if is_portrait:
            return resize_contain(
                product,
                size,
                background,
                padding_ratio=portrait_padding,
                template=category_template,
            )
        return resize_contain(product, size, background, padding_ratio=0.04, template=category_template)

    image = image.convert("RGB")
    if analysis.light_background:
        if analysis.touches_edge:
            return resize_cover(image, size)
        extension = render_rules["background_extension"]
        assert isinstance(extension, dict)
        if extension.get("enabled_for_opaque_main_product", True) and (
            is_portrait or category_template is not None or source_scale_adjustment != 0
        ):
            return _scaled_opaque_canvas(
                analysis,
                size,
                category_template,
                source_scale_adjustment,
                render_rules,
                portrait_padding,
            )
        return resize_contain(image.copy(), size, analysis.edge_color or background, padding_ratio=0.10)

    image_ratio = image.width / image.height
    target_ratio = size[0] / size[1]
    if image_ratio > 0.9 * target_ratio and image_ratio < 1.1 * target_ratio:
        return resize_cover(image, size)

    # Dark or saturated detail photos are meant to run edge-to-edge. Covering
    # the canvas preserves their aspect ratio while avoiding an added frame.
    return resize_cover(image, size)


def output_stem(source: Path) -> str:
    if source.suffix.lower() == ".png":
        return f"{source.stem}_png"
    if source.stem.endswith(" 2"):
        return source.stem.replace(" 2", "_2")
    return f"{source.stem}_jpg"


def _safe_filename_component(value: str, max_length: int = SHORT_NAME_MAX_LENGTH) -> str:
    """Keep generated names portable across Windows and macOS."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if not cleaned:
        cleaned = "素材"
    return cleaned[:max_length].rstrip(" .") or "素材"


def _short_group_label(value: str) -> str:
    """Keep the complete color name after a gender marker when available."""
    gender_match = re.search(r"男女|男|女", value)
    if gender_match:
        gender = gender_match.group(0)
        # Folder names normally follow "款号 性别 颜色".  The former compact
        # token rule lost names such as "摩卡棕" and collapsed them into 男_2.
        trailing_color = value[gender_match.end():].strip(" _-—·()（）[]【】")
        if trailing_color:
            return f"{gender}_{_safe_filename_component(trailing_color, 12)}"

    color_tokens = (
        "摩卡棕", "暗夜黑", "耐久灰", "云霜米", "卡其", "米白", "藏青",
        "深灰", "浅灰", "多色", "黑色", "白色", "红色", "蓝色", "绿色",
        "黄色", "灰色", "粉色", "紫色", "橙色", "棕色", "咖色",
        "黑", "白", "红", "蓝", "绿", "黄", "灰", "粉", "紫", "橙", "棕", "咖",
    )
    color = next((token for token in color_tokens if token in value), "")
    if gender_match and color:
        return f"{gender_match.group(0)}_{color}"
    if gender_match:
        return gender_match.group(0)
    if color:
        return color
    return _safe_filename_component(value, 8) if len(value) <= 8 else "素材"


def _build_short_source_names(
    sources: list[Path], product_root: Path
) -> tuple[dict[Path, str], dict[Path, str]]:
    """Create short group/source names while retaining identity in the manifest."""
    material_dirs = tuple(dict.fromkeys(source.parent for source in sources))
    group_names: dict[Path, str] = {}
    used_group_names: dict[str, int] = {}
    for material_dir in material_dirs:
        if material_dir == product_root:
            group_names[material_dir] = "素材"
            continue
        label = _short_group_label(material_dir.name)
        used_group_names[label] = used_group_names.get(label, 0) + 1
        suffix = "" if used_group_names[label] == 1 else f"_{used_group_names[label]}"
        group_names[material_dir] = f"{label}{suffix}"

    source_counts: dict[Path, int] = {}
    source_names: dict[Path, str] = {}
    for source in sources:
        source_counts[source.parent] = source_counts.get(source.parent, 0) + 1
        group_name = group_names[source.parent]
        if source_counts[source.parent] == 1 and sum(
            candidate.parent == source.parent for candidate in sources
        ) == 1:
            source_names[source] = group_name
        else:
            source_names[source] = f"{group_name}_{source_counts[source.parent]:02d}"
    return source_names, group_names


def _source_output_stem(
    source: Path, short_names: dict[Path, str] | None, naming_mode: str
) -> str:
    if naming_mode == "short" and short_names is not None:
        return short_names.get(source, f"素材_{source.stem[:SHORT_NAME_MAX_LENGTH]}")
    return output_stem(source)


def has_transparency(image: Image.Image) -> bool:
    if "A" not in image.getbands() and "transparency" not in image.info:
        return False
    return image.convert("RGBA").getchannel("A").getextrema()[0] < 255


def save_jpeg_with_limit(image: Image.Image, output_path: Path, max_size_mb: float | None) -> None:
    if max_size_mb is None:
        image.save(output_path, format="JPEG", quality=95, subsampling=0, optimize=True)
        return

    quality = 95
    while True:
        stream = BytesIO()
        image.save(stream, format="JPEG", quality=quality, subsampling=0, optimize=True)
        if stream.tell() <= max_size_mb * 1024 * 1024:
            output_path.write_bytes(stream.getvalue())
            return
        if quality > 45:
            quality -= 5
            continue
        output_path.write_bytes(stream.getvalue())
        return


def save_png_with_limit(image: Image.Image, output_path: Path, max_size_mb: float | None) -> None:
    if max_size_mb is None:
        image.save(output_path, format="PNG", optimize=True)
        return

    stream = BytesIO()
    image.save(stream, format="PNG", optimize=True, compress_level=9)
    output_path.write_bytes(stream.getvalue())


def _selling_point_font(size: int) -> ImageFont.ImageFont:
    """Use a bundled font when supplied; fall back safely during development."""
    font_path = RESOURCE_ROOT / "fonts" / "NotoSansSC-Bold.otf"
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size)
    if sys.platform == "darwin":
        system_font = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")
        if system_font.exists():
            return ImageFont.truetype(str(system_font), size, index=0)
    return ImageFont.load_default()


def render_selling_point_image(image: Image.Image, point: str) -> Image.Image:
    """Lay out one selling point without hiding the detail photo under a block."""
    canvas = image.convert("RGB").copy()
    width, height = canvas.size
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    alpha = Image.new("L", (1, height))
    fade_start = round(height * 0.58)
    alpha.putdata(
        [
            0
            if y < fade_start
            else round(205 * ((y - fade_start) / max(1, height - fade_start)) ** 0.72)
            for y in range(height)
        ]
    )
    shade = Image.new("RGBA", canvas.size, (8, 20, 32, 0))
    shade.putalpha(alpha.resize(canvas.size))
    overlay = Image.alpha_composite(overlay, shade)
    draw = ImageDraw.Draw(overlay)
    max_width = round(width * 0.78)
    font_size = max(24, round(width * 0.052))
    font = _selling_point_font(font_size)
    point_text = point.strip()
    lines: list[str] = []
    current = ""
    for character in point_text:
        candidate = current + character
        if current and draw.textlength(candidate, font=font) > max_width:
            lines.append(current)
            current = character
        else:
            current = candidate
    if current:
        lines.append(current)
    line_height = max(1, round(font_size * 1.3))
    while len(lines) > 2 and font_size > 24:
        font_size -= 2
        font = _selling_point_font(font_size)
        lines = []
        current = ""
        for character in point_text:
            candidate = current + character
            if current and draw.textlength(candidate, font=font) > max_width:
                lines.append(current)
                current = character
            else:
                current = candidate
        if current:
            lines.append(current)
        line_height = max(1, round(font_size * 1.3))
    x = round(width * 0.07)
    y = height - round(height * 0.09) - line_height * len(lines)
    for line in lines:
        draw.text((x + 2, y + 2), line, fill=(0, 0, 0, 110), font=font)
        draw.text((x, y), line, fill=(255, 255, 255, 255), font=font)
        y += line_height
    return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")


def generate_selling_point_images(
    sources: list[Path],
    decisions: dict[Path, SourceTypeDecision],
    output_root: Path,
    selected_sizes: tuple[str, ...],
    selling_points: tuple[str, ...],
    selling_point_matches: dict[Path, str] | None = None,
    notify_progress: Callable[[str], None] | None = None,
    max_size_mb: float | None = None,
    generated_paths: set[Path] | None = None,
    validation_issues: list[ValidationIssue] | None = None,
    failures: list[dict[str, str]] | None = None,
    short_names: dict[Path, str] | None = None,
    short_group_names: dict[Path, str] | None = None,
    naming_mode: str = "original",
    source_records: dict[Path, dict[str, object]] | None = None,
) -> int:
    """Assign configured selling points, using AI matches when available."""
    if not selling_points:
        return 0
    detail_sources = [source for source in sources if decisions.get(source, SourceTypeDecision("detail", 0, "", "")).source_type == "detail"]
    count = 0
    for index, source in enumerate(detail_sources):
        record = source_records.get(source) if source_records is not None else None
        if record is not None and record.get("reusable"):
            reused_paths = [
                Path(path)
                for path in record.get("paths", [])
                if isinstance(path, (str, Path))
            ]
            selling_paths = [path for path in reused_paths if "卖点图" in path.parts]
            expected = sum(1 for size_name in selected_sizes if size_name in SIZES)
            if expected and len(selling_paths) >= expected and all(path.is_file() for path in selling_paths):
                if generated_paths is not None:
                    generated_paths.update(selling_paths)
                record["selling_point_images"] = len(selling_paths)
                count += len(selling_paths)
                if notify_progress:
                    notify_progress(f"卖点图：{source.name}（复用）")
                continue
        point = (selling_point_matches or {}).get(
            source, selling_points[index % len(selling_points)]
        )
        try:
            with Image.open(source) as opened:
                image = opened.convert("RGB")
            for size_name, (size, _background) in SIZES.items():
                if size_name not in selected_sizes:
                    continue
                base = resize_cover(image, size)
                canvas = render_selling_point_image(base, point)
                group_name = (
                    short_group_names.get(source.parent, "素材")
                    if naming_mode == "short" and short_group_names is not None
                    else source.parent.name
                )
                output_dir = output_root / "卖点图" / group_name / size_name
                output_dir.mkdir(parents=True, exist_ok=True)
                short_point = _safe_filename_component(point, 12)
                output_path = output_dir / f"{_source_output_stem(source, short_names, naming_mode)}_{short_point}.jpg"
                save_jpeg_with_limit(canvas, output_path, max_size_mb)
                if generated_paths is not None:
                    generated_paths.add(output_path)
                if record is not None:
                    record.setdefault("paths", []).append(output_path)
                    record["selling_point_images"] = int(record.get("selling_point_images", 0) or 0) + 1
                if validation_issues is not None:
                    validation_issues.extend(validate_output(output_path, size, "RGB", max_size_mb))
                count += 1
                if notify_progress:
                    notify_progress(f"卖点图：{output_path.name}")
        except Exception as error:  # noqa: BLE001 - retain the remaining detail outputs
            if failures is not None:
                failures.append({"source": str(source), "code": "selling_point_output_failed", "message": str(error)})
    return count


def generate_transparent_products(
    product_root: Path,
    output_root: Path,
    notify_progress: Callable[[str], None] | None = None,
    max_size_mb: float | None = None,
    template_manager: CategoryTemplateManager | None = None,
    category: str | None = None,
    source_scale_adjustments: dict[str, float] | None = None,
    analysis_cache: dict[Path, SourceAnalysis] | None = None,
    generated_paths: set[Path] | None = None,
    validation_issues: list[ValidationIssue] | None = None,
    failures: list[dict[str, str]] | None = None,
    short_names: dict[Path, str] | None = None,
    short_group_names: dict[Path, str] | None = None,
    naming_mode: str = "original",
    source_records: dict[Path, dict[str, object]] | None = None,
    allowed_source_paths: set[Path] | None = None,
) -> int:
    count = 0
    for color_dir in _material_dirs(product_root):
        pngs = []
        for candidate in sorted(color_dir.glob("*.png")):
            try:
                with Image.open(candidate) as image:
                    if has_transparency(image):
                        pngs.append(candidate)
            except Exception as error:  # noqa: BLE001 - continue the batch
                if failures is not None:
                    failures.append({"source": str(candidate), "code": "invalid_image", "message": str(error)})
        if not pngs:
            continue
        for png in pngs:
            if allowed_source_paths is not None and png not in allowed_source_paths:
                continue
            try:
                record = source_records.get(png) if source_records is not None else None
                if record is not None and record.get("reusable"):
                    reused_paths = [
                        Path(path)
                        for path in record.get("paths", [])
                        if isinstance(path, (str, Path))
                    ]
                    transparent_paths = [path for path in reused_paths if "透明产品图" in path.parts]
                    if transparent_paths and all(path.is_file() for path in transparent_paths):
                        if generated_paths is not None:
                            generated_paths.update(transparent_paths)
                        count += len(transparent_paths)
                        if notify_progress:
                            notify_progress(f"透明图：{png.name}（复用）")
                        continue
                analysis = analysis_cache.get(png) if analysis_cache is not None else None
                if analysis is None:
                    analysis = analyze_source(png)
                    if analysis_cache is not None:
                        analysis_cache[png] = analysis
                image = analysis.image.convert("RGBA")
                adjustment = float(_source_mapping_value(png, source_scale_adjustments, 0.0))
                for size_name, size in TRANSPARENT_SIZES.items():
                    if template_manager is not None and category is not None:
                        template = template_manager.get(
                            category, TRANSPARENT_TEMPLATE_SIZES[size_name]
                        )
                        template = replace(template, scale=template.scale * (1 + adjustment / 100))
                        canvas = resize_transparent_with_template(image, size, template)
                    else:
                        working = image.copy()
                        target_edge = max(1, round(1325 * (1 + adjustment / 100)))
                        working.thumbnail((target_edge, target_edge), Image.Resampling.LANCZOS)
                        canvas = Image.new("RGBA", size, (0, 0, 0, 0))
                        position = ((size[0] - working.width) // 2, (size[1] - working.height) // 2)
                        canvas.alpha_composite(working, position)
                    output_dir = output_root / "透明产品图" / size_name
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"{_source_output_stem(png, short_names, naming_mode)}_透明.png"
                    save_png_with_limit(canvas, output_path, max_size_mb)
                    if generated_paths is not None:
                        generated_paths.add(output_path)
                    if record is not None:
                        record.setdefault("paths", []).append(output_path)
                        record["transparent_images"] = int(record.get("transparent_images", 0) or 0) + 1
                    if validation_issues is not None:
                        validation_issues.extend(validate_output(output_path, size, "RGBA", max_size_mb))
                    count += 1
                    if notify_progress:
                        notify_progress(f"透明图：{output_path.name}")
            except Exception as error:  # noqa: BLE001 - keep the remaining sources running
                if failures is not None:
                    failures.append({"source": str(png), "code": "transparent_output_failed", "message": str(error)})
    return count


def cleanup_stale_transparent_products(
    product_root: Path,
    output_root: Path,
    transparent_sources: set[Path],
) -> int:
    removed = 0
    for color_dir in _material_dirs(product_root):
        for png in sorted(color_dir.glob("*.png")):
            if png in transparent_sources:
                continue
            for size_name in TRANSPARENT_SIZES:
                stale_path = output_root / "透明产品图" / size_name / f"{color_dir.name}_{png.stem}_透明.png"
                if stale_path.exists():
                    stale_path.unlink()
                    removed += 1
    return removed


MANIFEST_NAME = ".pixelflow-manifest.json"
MANIFEST_TTL_DAYS = 7
MANIFEST_CACHE_DIR_NAME = "PixelFlow/manifests"


def _manifest_paths(output_root: Path) -> set[str]:
    payload = _read_manifest(output_root)
    paths = payload.get("generated_paths", [])
    return {str(path) for path in paths if isinstance(path, str)}


def _manifest_cache_path(output_root: Path) -> Path:
    """Return a private cache path keyed by the output folder location."""
    resolved_root = output_root.expanduser().resolve(strict=False)
    if sys.platform == "darwin":
        cache_root = Path.home() / "Library" / "Caches"
    else:
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    key = hashlib.sha256(str(resolved_root).encode("utf-8")).hexdigest()
    return cache_root / MANIFEST_CACHE_DIR_NAME / f"{key}.json"


def _manifest_is_fresh(payload: dict[str, object]) -> bool:
    generated_at = payload.get("generated_at")
    try:
        created = datetime.fromisoformat(str(generated_at))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_seconds = (datetime.now(timezone.utc) - created).total_seconds()
    except (TypeError, ValueError):
        # Preserve compatibility with older or manually edited manifests;
        # parameter/version checks still decide whether they can be reused.
        return True
    return age_seconds <= MANIFEST_TTL_DAYS * 24 * 60 * 60


def _read_manifest_file(manifest_path: Path) -> dict[str, object]:
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_manifest(output_root: Path) -> dict[str, object]:
    cache_path = _manifest_cache_path(output_root)
    payload = _read_manifest_file(cache_path)
    if payload and not _manifest_is_fresh(payload):
        try:
            cache_path.unlink()
        except OSError:
            pass
        return {}
    if payload:
        return payload

    # Migrate manifests created by older builds. The output folder remains
    # clean after this read, while existing incremental records are preserved.
    legacy_path = output_root / MANIFEST_NAME
    legacy_payload = _read_manifest_file(legacy_path)
    if not legacy_payload:
        return {}
    if not _manifest_is_fresh(legacy_payload):
        try:
            legacy_path.unlink()
        except OSError:
            pass
        return {}
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(legacy_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        legacy_path.unlink()
    except OSError:
        # Read-only/shared output folders can still be used; just leave the
        # legacy file in place when migration is not permitted.
        pass
    return legacy_payload


def _source_signatures(product_root: Path, sources: list[Path]) -> dict[str, dict[str, int]]:
    signatures: dict[str, dict[str, int]] = {}
    for source in sources:
        try:
            stat = source.stat()
        except OSError:
            continue
        signatures[source.relative_to(product_root).as_posix()] = {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    return signatures


def _manifest_matches(
    output_root: Path,
    parameters: dict[str, object],
    source_signatures: dict[str, dict[str, int]],
) -> dict[str, object] | None:
    """Return a manifest when all inputs and settings still match the last run."""
    payload = _read_manifest(output_root)
    if not payload:
        return None
    if payload.get("parameters") != parameters:
        return None
    if payload.get("source_signatures") != source_signatures:
        return None
    paths = payload.get("generated_paths", [])
    if not isinstance(paths, list) or not paths:
        return None
    for relative in paths:
        if not isinstance(relative, str):
            return None
        candidate = output_root / relative
        try:
            candidate.relative_to(output_root)
        except ValueError:
            return None
        if not candidate.is_file():
            return None
    return payload


def _write_manifest(
    output_root: Path,
    generated_paths: set[Path],
    parameters: dict[str, object],
    source_mappings: list[dict[str, str]] | None = None,
    source_signatures: dict[str, dict[str, int]] | None = None,
    result_summary: dict[str, int] | None = None,
    source_records: dict[str, dict[str, object]] | None = None,
) -> None:
    payload = {
        "pixel_flow_version": ENGINE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": parameters,
        "generated_paths": sorted(path.relative_to(output_root).as_posix() for path in generated_paths),
        "source_mappings": source_mappings or [],
        "source_signatures": source_signatures or {},
        "result_summary": result_summary or {},
        "source_records": source_records or {},
    }
    manifest_path = _manifest_cache_path(output_root)
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        # Manifest is an optional performance cache; a cache-directory
        # permission problem must not turn a completed generation into a
        # failed task.
        pass


def _cleanup_manifest_outputs(output_root: Path, generated_paths: set[Path]) -> int:
    current = {path.relative_to(output_root).as_posix() for path in generated_paths}
    removed = 0
    for relative in _manifest_paths(output_root) - current:
        candidate = output_root / relative
        try:
            candidate.relative_to(output_root)
        except ValueError:
            continue
        if candidate.is_file():
            candidate.unlink()
            removed += 1
    return removed


def _source_mapping_value(source: Path, values: dict[str, object] | None, default: object) -> object:
    if not values:
        return default
    for key in (str(source), source.as_posix(), source.name):
        if key in values:
            return values[key]
    return default


def generate_vip_square_images(
    output_root: Path,
    border_path: Path,
    notify_progress: Callable[[str], None] | None = None,
    max_size_mb: float | None = None,
    allowed_source_paths: set[Path] | None = None,
    generated_paths: set[Path] | None = None,
    validation_issues: list[ValidationIssue] | None = None,
    failures: list[dict[str, str]] | None = None,
    reusable_paths: set[Path] | None = None,
    source_records: dict[Path, dict[str, object]] | None = None,
) -> int:
    if not border_path.exists():
        raise FileNotFoundError(f"唯品会边框不存在：{border_path}")
    border = Image.open(border_path).convert("RGBA")
    count = 0
    for variant_dir in sorted(output_root.iterdir()):
        if not variant_dir.is_dir() or variant_dir.name in {"透明产品图", "唯品专享1440"}:
            continue
        source_root = variant_dir / "1440x1440"
        if not source_root.is_dir():
            continue
        for logo_state in ("无Logo",):
            output_dir = variant_dir / "唯品专享1440" / logo_state
            output_dir.mkdir(parents=True, exist_ok=True)
            for source in sorted((source_root / logo_state).glob("*.jpg")):
                if allowed_source_paths is not None and source not in allowed_source_paths:
                    continue
                try:
                    with Image.open(source) as opened:
                        image = opened.convert("RGBA")
                    image = Image.alpha_composite(image, border)
                    output_path = output_dir / source.name
                    if reusable_paths is not None and output_path in reusable_paths:
                        if generated_paths is not None:
                            generated_paths.add(output_path)
                        count += 1
                        if notify_progress:
                            notify_progress(f"唯品图：{output_path.name}（复用）")
                        continue
                    save_jpeg_with_limit(image.convert("RGB"), output_path, max_size_mb)
                    if generated_paths is not None:
                        generated_paths.add(output_path)
                    if validation_issues is not None:
                        validation_issues.extend(validate_output(output_path, (1440, 1440), "RGB", max_size_mb))
                    if source_records is not None:
                        for record in source_records.values():
                            if source in record.get("paths", []):
                                record["paths"].append(output_path)
                                record["vip_images"] = int(record.get("vip_images", 0) or 0) + 1
                                break
                    count += 1
                    if notify_progress:
                        notify_progress(f"唯品图：{output_path.name}")
                except Exception as error:  # noqa: BLE001 - keep the remaining sources running
                    if failures is not None:
                        retry_source = None
                        if source_records is not None:
                            for original, record in source_records.items():
                                if source in record.get("paths", []):
                                    retry_source = str(original)
                                    break
                        failure = {"source": str(source), "code": "vip_output_failed", "message": str(error)}
                        if retry_source:
                            failure["retry_source"] = retry_source
                        failures.append(failure)
    return count


def generate_images(
    product_root: Path,
    output_root: Path,
    resource_root: Path = RESOURCE_ROOT,
    selected_sizes: tuple[str, ...] = tuple(SIZES),
    include_logo: bool = True,
    include_vip: bool = True,
    progress: ProgressCallback | None = None,
    max_size_mb: float | None = None,
    category_override: str | None = None,
    enable_category_template: bool = False,
    ai_category_provider: AICategoryProvider | None = None,
    enable_material_understanding: bool = False,
    source_type_overrides: dict[str, str] | None = None,
    excluded_model_sources: set[str] | None = None,
    ai_source_type_provider: AISourceTypeProvider | None = None,
    source_scale_adjustments: dict[str, float] | None = None,
    folder_category_overrides: dict[str, str] | None = None,
    include_model_images: bool = True,
    include_selling_point_images: bool = False,
    selling_points: tuple[str, ...] = (),
    ai_assist: bool = False,
    ai_api_key: str | None = None,
    ai_model: str | None = None,
    logo_overrides: dict[str, Path] | None = None,
    cancel_event: object | None = None,
    naming_mode: str = "short",
    local_model_enabled: bool = False,
    local_model_path: str | Path | None = None,
    source_filter: set[str] | None = None,
) -> dict[str, object]:
    logo_paths = {
        "logo": resource_root / "新logo 1440.png",
        "white_logo": resource_root / "新logo 1440 2.png",
        "tall_logo": resource_root / "新logo 1920.png",
        "tall_white_logo": resource_root / "新logo 1920 2.png",
        "border": resource_root / "唯品会边框.png",
    }
    for key, path in (logo_overrides or {}).items():
        if key in logo_paths and Path(path).is_file():
            logo_paths[key] = Path(path)
    required_resources = list(logo_paths.values()) if include_logo else []
    if include_vip and "1440x1440" in selected_sizes:
        required_resources.append(logo_paths["border"])
    for resource_path in required_resources:
        if not resource_path.exists():
            raise FileNotFoundError(f"资源文件不存在：{resource_path}")
    if not product_root.is_dir():
        raise FileNotFoundError(f"产品素材文件夹不存在：{product_root}")
    if not selected_sizes:
        raise ValueError("至少选择一个输出尺寸")
    material_dirs = _material_dirs(product_root)
    if not material_dirs:
        raise ValueError("源素材文件夹内没有图片或有效的素材子文件夹")

    logo = Image.open(logo_paths["logo"]).convert("RGBA") if include_logo else None
    white_logo = Image.open(logo_paths["white_logo"]).convert("RGBA") if include_logo else None
    tall_logo = Image.open(logo_paths["tall_logo"]).convert("RGBA") if include_logo else None
    tall_white_logo = Image.open(logo_paths["tall_white_logo"]).convert("RGBA") if include_logo else None
    all_sources = sorted(
        path
        for color_dir in material_dirs
        for path in sorted(color_dir.iterdir())
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    source_filter = source_filter or None

    def matches_source_filter(source: Path) -> bool:
        # Retry filters represent concrete input files.  Do not fall back to
        # basename matching here: duplicate names in different color folders
        # must not cause more than the requested item to be regenerated.
        return source_filter is None or any(
            key in source_filter for key in (str(source), source.as_posix())
        )

    sources = [
        source
        for source in all_sources
        if matches_source_filter(source)
    ]
    if not sources:
        raise ValueError("素材文件夹内没有 JPG、JPEG 或 PNG 图片")
    naming_mode = naming_mode if naming_mode in {"short", "original"} else "short"
    # Build names from the complete input set so a retry-only run produces
    # exactly the same output paths as the original batch.
    short_names, short_group_names = _build_short_source_names(all_sources, product_root)
    source_mappings = [
        {
            "short_name": short_names[source],
            "output_stem": _source_output_stem(source, short_names, naming_mode),
            "original_name": source.name,
            "original_path": source.relative_to(product_root).as_posix(),
        }
        for source in sources
    ]

    ai_provider = (
        OpenAIVisionAssistant.from_environment(ai_api_key, ai_model)
        if ai_assist
        else None
    )
    if ai_assist and ai_provider is None:
        raise ValueError(
            "已启用 AI 辅助，但未找到 OPENAI_API_KEY。请在启动 App 前配置环境变量。"
        )

    category_decision = None
    template_manager = None
    if enable_category_template:
        category_decision = RuleCategoryClassifier(ai_category_provider).classify(
            product_root,
            manual_category=category_override,
        )
        template_manager = CategoryTemplateManager(
            resource_root / "templates" / "category_templates.json"
        )
    model_product_category = (
        category_decision.category if category_decision is not None else category_override
    )
    manifest_parameters: dict[str, object] = {
        "engine_version": ENGINE_VERSION,
        "selected_sizes": list(selected_sizes),
        "include_logo": include_logo,
        "include_vip": include_vip,
        "include_model_images": include_model_images,
        "include_selling_point_images": include_selling_point_images,
        "selling_points": list(selling_points),
        "max_size_mb": max_size_mb,
        "category": category_decision.category if category_decision is not None else None,
        "naming_mode": naming_mode,
        "local_model_enabled": local_model_enabled,
        "local_model_path": str(local_model_path or ""),
        "source_type_overrides": source_type_overrides or {},
        "excluded_model_sources": sorted(excluded_model_sources or set()),
        "source_scale_adjustments": source_scale_adjustments or {},
        "folder_category_overrides": folder_category_overrides or {},
        "logo_overrides": {
            key: str(value) for key, value in (logo_overrides or {}).items()
        },
    }
    source_signatures = _source_signatures(product_root, sources)
    previous_manifest = _read_manifest(output_root)
    previous_records_raw = previous_manifest.get("source_records", {})
    reusable_records: dict[Path, dict[str, object]] = {}
    if (
        source_filter is None
        and not ai_assist
        and previous_manifest.get("parameters") == manifest_parameters
        and isinstance(previous_records_raw, dict)
    ):
        for source in sources:
            relative = source.relative_to(product_root).as_posix()
            raw_record = previous_records_raw.get(relative)
            if not isinstance(raw_record, dict):
                continue
            if raw_record.get("signature") != source_signatures.get(relative):
                continue
            raw_paths = raw_record.get("paths", [])
            if not isinstance(raw_paths, list) or not raw_paths:
                continue
            paths: list[Path] = []
            valid = True
            for raw_path in raw_paths:
                if not isinstance(raw_path, str):
                    valid = False
                    break
                candidate = output_root / raw_path
                try:
                    candidate.relative_to(output_root)
                except ValueError:
                    valid = False
                    break
                if not candidate.is_file():
                    valid = False
                    break
                paths.append(candidate)
            if valid:
                reusable_records[source] = {
                    **raw_record,
                    "paths": paths,
                    "reusable": True,
                }
    # AI-assisted classification can legitimately vary between runs, so do
    # not reuse a previous batch when the remote assistant is enabled.
    if not ai_assist and source_filter is None:
        cached_manifest = _manifest_matches(
            output_root,
            manifest_parameters,
            source_signatures,
        )
        if cached_manifest is not None:
            summary = cached_manifest.get("result_summary")
            summary = summary if isinstance(summary, dict) else {}
            return {
                "sources": len(sources),
                "main_images": int(summary.get("main_images", 0) or 0),
                "model_images": int(summary.get("model_images", 0) or 0),
                "selling_point_images": int(summary.get("selling_point_images", 0) or 0),
                "transparent_images": int(summary.get("transparent_images", 0) or 0),
                "vip_images": int(summary.get("vip_images", 0) or 0),
                "render_workers": int(summary.get("render_workers", 1) or 1),
                "local_model_used": bool(summary.get("local_model_used", False)),
                "local_model_errors": [],
                "cancelled": False,
                "incremental_skipped": True,
                "stale_outputs_removed": 0,
                "failures": [],
                "validation": {"passed": len(cached_manifest.get("generated_paths", [])), "warnings": [], "failed": []},
                "category": manifest_parameters.get("category"),
                "category_source": "manifest",
                "category_confidence": 1.0,
                "output_root": str(output_root),
            }
    render_rules = load_render_rules(resource_root)

    source_type_decisions: dict[Path, SourceTypeDecision] = {}
    if enable_material_understanding or include_model_images:
        source_type_provider = ai_source_type_provider or ai_provider
        source_type_decisions = RuleSourceTypeClassifier(
            source_type_provider,
            prefer_ai=ai_provider is not None,
            ai_context={
                "category": category_decision.category if category_decision else None,
                "template_scale": (
                    template_manager.get(category_decision.category, "1440x1440").scale
                    if template_manager is not None and category_decision is not None
                    else None
                ),
            },
        ).classify_many(
            sources,
            overrides=source_type_overrides,
        )

    def is_model_source_excluded(
        source: Path,
        decision: SourceTypeDecision | None,
    ) -> bool:
        """Skip model material explicitly marked as irrelevant to this batch."""
        if decision is None or decision.source_type not in {"model", "model_detail"}:
            return False
        excluded = excluded_model_sources or set()
        return any(key in excluded for key in (str(source), source.as_posix(), source.name))

    local_model_assistant: LocalModelAssistant | None = None
    local_model_errors: list[str] = []
    if local_model_enabled:
        model_path = find_compatible_onnx_model(
            local_model_path or "",
            RESOURCE_ROOT / "models",
        )
        if model_path is None:
            local_model_errors.append("未找到 ONNX 模型文件，已回退规则构图")
        else:
            try:
                local_model_assistant = LocalModelAssistant(model_path)
            except Exception as error:  # noqa: BLE001 - model is an optional enhancement
                local_model_errors.append(f"本地模型加载失败：{error}")

    # The confirmation dialog may have already used the pose model to correct
    # a saturated-background photo, but generation classifies the sources
    # independently.  Apply the same correction here or the worker would
    # silently turn those model photos back into detail images.
    if local_model_assistant is not None:
        for source in sources:
            override_value = next(
                (
                    (source_type_overrides or {}).get(key)
                    for key in (str(source), source.name, source.as_posix())
                    if (source_type_overrides or {}).get(key)
                ),
                None,
            )
            if override_value in {"detail", "model_detail"}:
                continue
            try:
                composition = local_model_assistant.compose_for_sizes(
                    source,
                    SIZES,
                    product_category=model_product_category,
                )
                model_view = composition.get("model_view")
                confidence = float(composition.get("confidence", 0.0))
                rule_decision = source_type_decisions[source]
                can_promote_model = (
                    rule_decision.source_type in {"model", "model_detail"}
                    or model_view == "group"
                )
                credible_model = credible_model_composition(composition)
                if model_view in {"full_body", "upper_body", "lower_body", "group"} and can_promote_model and credible_model:
                    source_type_decisions[source] = SourceTypeDecision(
                        "model",
                        confidence,
                        "local_onnx",
                        "本地模型检测到人物主体"
                        + ("（多人）" if model_view == "group" else ""),
                        metadata=composition,
                    )
                elif source_type_decisions[source].source_type == "model" and credible_model:
                    source_type_decisions[source] = SourceTypeDecision(
                        "model_detail",
                        confidence,
                        "local_onnx",
                        "本地模型识别为模特细节图，按细节图铺满处理",
                        metadata=composition,
                    )
                elif (
                    source_type_decisions[source].source_type in {"model", "model_detail"}
                    and source_type_decisions[source].source != "manual"
                    and not credible_model
                ):
                    source_type_decisions[source] = SourceTypeDecision(
                        "detail",
                        confidence,
                        "local_onnx",
                        "人体关键点/人物范围不足，按细节图处理",
                        metadata=composition,
                    )
            except Exception as error:  # noqa: BLE001 - preserve rule fallback
                local_model_errors.append(f"{source.name}：模型识别失败，已回退规则分类：{error}")

    transparent_sources: list[Path] = []
    failures: list[dict[str, str]] = []
    for color_dir in material_dirs:
        for candidate in sorted(color_dir.glob("*.png")):
            if not matches_source_filter(candidate):
                continue
            try:
                with Image.open(candidate) as image:
                    if has_transparency(image):
                        transparent_sources.append(candidate)
            except Exception as error:  # noqa: BLE001 - a damaged source must not end the batch
                failures.append({"source": str(candidate), "code": "invalid_image", "message": str(error)})
    main_sources = [
        source
        for source in sources
        if (
            source_type_decisions.get(
                source, SourceTypeDecision("main_product", 1.0, "default", "")
            ).source_type not in {"model", "model_detail"}
            or include_model_images
        )
        and not is_model_source_excluded(
            source,
            source_type_decisions.get(source),
        )
    ]
    main_total = len(main_sources) * len(selected_sizes) * (2 if include_logo else 1)
    transparent_total = len(transparent_sources) * len(TRANSPARENT_SIZES)
    vip_total = len(main_sources) if include_vip and "1440x1440" in selected_sizes else 0
    analysis_total = len(sources) if (enable_material_understanding or include_model_images) else 0
    detail_total = sum(
        decision.source_type == "detail" for decision in source_type_decisions.values()
    )
    ai_match_total = (
        detail_total
        if ai_provider is not None and include_selling_point_images and selling_points
        else 0
    )
    selling_point_total = (
        detail_total * len(selected_sizes)
        if include_selling_point_images and selling_points
        else 0
    )
    model_composition_sources = (
        [
            source
            for source in main_sources
            if source_type_decisions.get(source) is not None
            and source_type_decisions[source].source_type == "model"
        ]
        if local_model_assistant is not None
        else []
    )
    model_composition_total = len(model_composition_sources)
    total_steps = (
        analysis_total
        + ai_match_total
        + model_composition_total
        + main_total
        + transparent_total
        + vip_total
        + selling_point_total
    )
    completed_steps = 0
    progress_lock = Lock()

    def report(message: str) -> None:
        nonlocal completed_steps
        with progress_lock:
            completed_steps += 1
            if progress:
                progress(completed_steps, total_steps, message)

    if local_model_assistant is not None and model_composition_sources:
        def prepare_model_composition(source: Path):
            try:
                return (
                    source,
                    local_model_assistant.compose_for_sizes(
                        source,
                        SIZES,
                        product_category=model_product_category,
                    ),
                    None,
                )
            except Exception as error:  # noqa: BLE001 - one bad image must not stop the batch
                return source, None, str(error)

        model_workers = _auto_worker_count(len(model_composition_sources))
        with ThreadPoolExecutor(
            max_workers=model_workers,
            thread_name_prefix="PixelFlowModel",
        ) as executor:
            futures = [executor.submit(prepare_model_composition, source) for source in model_composition_sources]
            for future in as_completed(futures):
                source, composition, error = future.result()
                decision = source_type_decisions[source]
                if error:
                    local_model_errors.append(f"{source.name}：模型推理失败：{error}")
                elif composition is None:
                    local_model_errors.append(f"{source.name}：未得到唯一可信人物构图，已回退安全构图")
                elif composition.get("model_view") == "detail":
                    metadata = dict(decision.metadata or {})
                    metadata.update(composition)
                    source_type_decisions[source] = replace(
                        decision,
                        source_type="model_detail",
                        source="local_onnx",
                        reason="人体关键点不足，按模特细节图铺满处理",
                        metadata=metadata,
                    )
                else:
                    metadata = dict(decision.metadata or {})
                    metadata.update(composition)
                    source_type_decisions[source] = replace(
                        decision,
                        source="local_onnx",
                        metadata=metadata,
                    )
                report(f"本地构图判断：{source.name}")

    generated_paths: set[Path] = set()
    validation_issues: list[ValidationIssue] = []
    vip_source_paths: set[Path] = set()
    analysis_cache: dict[Path, SourceAnalysis] = {}
    selling_point_matches: dict[Path, str] = {}
    source_records_by_path: dict[Path, dict[str, object]] = {}

    if ai_provider is not None and include_selling_point_images and selling_points:
        for source in sources:
            decision = source_type_decisions.get(source)
            if decision is None or decision.source_type != "detail":
                continue
            matched = ai_provider.match_selling_point(source, selling_points)
            if matched:
                selling_point_matches[source] = matched
            report(f"AI卖点匹配：{source.name} → {matched or '规则回退'}")

    def render_source(source: Path) -> dict[str, object]:
        """Render one source independently so several sources can run safely."""
        reusable_record = reusable_records.get(source)
        if reusable_record is not None:
            # Manifest paths are serialized as strings; normalize them back to
            # Path objects before inspecting or returning them.
            paths = {
                path
                for raw_path in reusable_record.get("paths", [])
                if isinstance(raw_path, (str, Path))
                for raw in [Path(raw_path)]
                for path in [raw if raw.is_absolute() else output_root / raw]
            }
            for path in sorted(paths):
                if "透明产品图" not in path.parts and "唯品专享1440" not in path.parts and "卖点图" not in path.parts:
                    report(f"主图：{path.name}（复用）")
            return {
                "analysis": None,
                "paths": paths,
                "validation": [],
                "vip_paths": {
                    path for path in paths
                    if path.name.lower().endswith(".jpg") and "/1440x1440/无Logo/" in f"/{path.as_posix()}"
                },
                "failures": [],
                "main_count": int(reusable_record.get("main_images", 0) or 0),
                "model_count": int(reusable_record.get("model_images", 0) or 0),
                "record": reusable_record,
            }
        local_paths: set[Path] = set()
        local_validation: list[ValidationIssue] = []
        local_vip_paths: set[Path] = set()
        local_failures: list[dict[str, str]] = []
        local_main_count = 0
        local_model_count = 0
        analysis: SourceAnalysis | None = None
        try:
            if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                return {
                    "analysis": None,
                    "paths": local_paths,
                    "validation": local_validation,
                    "vip_paths": local_vip_paths,
                    "failures": local_failures,
                    "main_count": 0,
                    "model_count": 0,
                    "record": {"paths": [], "main_images": 0, "model_images": 0, "transparent_images": 0, "vip_images": 0, "selling_point_images": 0},
                }
            analysis = analyze_source(source)
            source_decision = source_type_decisions.get(source)
            if source_decision is not None:
                report(f"素材识别：{source.name} → {source_decision.source_type}")
            if is_model_source_excluded(source, source_decision):
                return {
                    "analysis": analysis,
                    "paths": local_paths,
                    "validation": local_validation,
                    "vip_paths": local_vip_paths,
                    "failures": local_failures,
                    "main_count": 0,
                    "model_count": 0,
                    "record": {"paths": [], "main_images": 0, "model_images": 0, "transparent_images": 0, "vip_images": 0, "selling_point_images": 0},
                }
            color_name = source.parent.name
            output_group_name = (
                short_group_names.get(source.parent, "素材")
                if naming_mode == "short"
                else color_name
            )
            if source_decision is not None and source_decision.source_type in {"model", "model_detail"} and not include_model_images:
                return {
                    "analysis": analysis,
                    "paths": local_paths,
                    "validation": local_validation,
                    "vip_paths": local_vip_paths,
                    "failures": local_failures,
                    "main_count": 0,
                    "model_count": 0,
                    "record": {"paths": [], "main_images": 0, "model_images": 0, "transparent_images": 0, "vip_images": 0, "selling_point_images": 0},
                }
            adjustment = float(_source_mapping_value(source, source_scale_adjustments, 0.0))
            adjustment = max(-50.0, min(50.0, adjustment))
            source_category = (
                str(_source_mapping_value(source.parent, folder_category_overrides, category_decision.category))
                if category_decision is not None
                else None
            )
            for size_name, (size, background) in SIZES.items():
                if size_name not in selected_sizes:
                    continue
                if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                    break
                category_template = (
                    template_manager.get(source_category, size_name)
                    if template_manager is not None and source_category is not None
                    and (source_decision is None or source_decision.source_type == "main_product")
                    else None
                )
                base = build_canvas(
                    source,
                    size,
                    background,
                    category_template,
                    source_type=source_decision.source_type if source_decision else None,
                    source_scale_adjustment=adjustment if source_decision is None or source_decision.source_type == "main_product" else 0.0,
                    analysis=analysis,
                    render_rules=render_rules,
                    model_composition=(source_decision.metadata if source_decision else None),
                )
                logo_source = logo if size_name == "1440x1440" else tall_logo
                white_logo_source = white_logo if size_name == "1440x1440" else tall_white_logo
                if include_logo:
                    logo_for_size = logo_source if logo_source.size == size else logo_source.resize(size, Image.Resampling.LANCZOS)
                    white_logo_for_size = white_logo_source if white_logo_source.size == size else white_logo_source.resize(size, Image.Resampling.LANCZOS)
                    logo_box = logo_for_size.getchannel("A").getbbox()
                else:
                    logo_for_size = white_logo_for_size = logo_box = None
                logo_states = ("含Logo", "无Logo") if include_logo else ("无Logo",)
                for logo_state in logo_states:
                    if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                        break
                    canvas = base.copy().convert("RGBA")
                    if include_logo and logo_state == "含Logo":
                        logo_area = base.crop(logo_box).convert("RGB")
                        mean_rgb = ImageStat.Stat(logo_area).mean
                        luminance = (
                            mean_rgb[0] * 0.2126
                            + mean_rgb[1] * 0.7152
                            + mean_rgb[2] * 0.0722
                        )
                        canvas.alpha_composite(
                            white_logo_for_size if luminance < 160 else logo_for_size,
                            (0, 0),
                        )
                    if source_decision is not None and source_decision.source_type in {"model", "model_detail"}:
                        model_group_names = {"模特图", "模特细节图", "模特主图", "model", "lookbook"}
                        model_variant = color_name.strip()
                        # Full-body and clothing-detail model shots share one
                        # output root.  Their internal types still control the
                        # crop rule, but the user no longer needs to browse
                        # two parallel model-image folder trees.
                        output_dir = output_root / "模特图"
                        if model_variant.casefold() not in model_group_names:
                            output_dir /= output_group_name
                        output_dir = output_dir / size_name / logo_state
                    else:
                        # A model-root batch can contain a close-up that the
                        # pose model safely downgrades to ``detail``. Keep it
                        # with the model batch instead of creating a stray
                        # output_root/素材 folder.
                        model_root = color_name.casefold() in MODEL_GROUP_NAMES
                        output_dir = output_root / ("模特图" if model_root else output_group_name) / size_name / logo_state
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"{_source_output_stem(source, short_names, naming_mode)}.jpg"
                    save_jpeg_with_limit(canvas.convert("RGB"), output_path, max_size_mb)
                    local_paths.add(output_path)
                    local_validation.extend(validate_output(output_path, size, "RGB", max_size_mb))
                    if size_name == "1440x1440" and logo_state == "无Logo":
                        local_vip_paths.add(output_path)
                    local_main_count += 1
                    # Keep a per-input output index for future incremental runs.
                    if source_decision is not None and source_decision.source_type in {"model", "model_detail"}:
                        local_model_count += 1
                    report(f"主图：{output_path.name}")
        except Exception as error:  # noqa: BLE001 - isolate a single bad source
            local_failures.append({"source": str(source), "code": "source_output_failed", "message": str(error)})
        return {
            "analysis": analysis,
            "paths": local_paths,
            "validation": local_validation,
            "vip_paths": local_vip_paths,
            "failures": local_failures,
            "main_count": local_main_count,
            "model_count": local_model_count,
            "record": {
                "paths": list(local_paths),
                "main_images": local_main_count,
                "model_images": local_model_count,
                "transparent_images": 0,
                "vip_images": 0,
                "selling_point_images": 0,
            },
        }

    main_count = 0
    model_count = 0
    cancelled = False
    worker_count = _auto_worker_count(len(main_sources))
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="PixelFlowRender") as executor:
        future_to_source = {
            executor.submit(render_source, source): source
            for source in main_sources
        }
        for future in as_completed(future_to_source):
            result = future.result()
            source = future_to_source[future]
            analysis = result["analysis"]
            if isinstance(analysis, SourceAnalysis):
                analysis_cache[source] = analysis
            generated_paths.update(result["paths"])
            validation_issues.extend(result["validation"])
            vip_source_paths.update(result["vip_paths"])
            failures.extend(result["failures"])
            main_count += int(result["main_count"])
            model_count += int(result["model_count"])
            source_records_by_path[source] = dict(result.get("record", {}))
    cancelled = bool(cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)())

    # Keep a record slot for sources that do not produce a main/model image
    # (for example an excluded model photo) but may still produce a transparent
    # product image.  This lets the next incremental run reuse that output too.
    for source in sources:
        source_records_by_path.setdefault(
            source,
            {
                "paths": [],
                "main_images": 0,
                "model_images": 0,
                "transparent_images": 0,
                "vip_images": 0,
                "selling_point_images": 0,
            },
        )

    transparent_count = 0
    vip_count = 0
    selling_point_count = 0
    if not cancelled:
        transparent_count = generate_transparent_products(
            product_root,
            output_root,
            report,
            max_size_mb,
            template_manager,
            category_decision.category if category_decision is not None else None,
            source_scale_adjustments,
            analysis_cache,
            generated_paths,
            validation_issues,
            failures,
            short_names,
            short_group_names,
            naming_mode,
            source_records_by_path,
            set(sources),
        )
        vip_count = (
            generate_vip_square_images(
                output_root,
                logo_paths["border"],
                report,
                max_size_mb,
                vip_source_paths,
                generated_paths,
                validation_issues,
                failures,
                {
                    path
                    for record in reusable_records.values()
                    for path in record.get("paths", [])
                    if isinstance(path, Path) and "唯品专享1440" in path.parts
                },
                source_records_by_path,
            )
            if include_vip and "1440x1440" in selected_sizes
            else 0
        )
        if include_selling_point_images:
            selling_point_count = generate_selling_point_images(
                sources,
                source_type_decisions,
                output_root,
                selected_sizes,
                selling_points,
                selling_point_matches,
                report,
                max_size_mb,
                generated_paths,
                validation_issues,
                failures,
                short_names,
                short_group_names,
                naming_mode,
                source_records_by_path,
            )
    stale_removed = 0
    if not cancelled and not failures and source_filter is None:
        stale_removed = _cleanup_manifest_outputs(output_root, generated_paths)
        serialized_source_records: dict[str, dict[str, object]] = {}
        for source, record in source_records_by_path.items():
            relative = source.relative_to(product_root).as_posix()
            serialized_source_records[relative] = {
                "signature": source_signatures.get(relative),
                "paths": sorted(
                    path.relative_to(output_root).as_posix()
                    for path in record.get("paths", [])
                    if isinstance(path, Path)
                    and path.exists()
                    and path.is_relative_to(output_root)
                ),
                "main_images": int(record.get("main_images", 0) or 0),
                "model_images": int(record.get("model_images", 0) or 0),
                "transparent_images": int(record.get("transparent_images", 0) or 0),
                "vip_images": int(record.get("vip_images", 0) or 0),
                "selling_point_images": int(record.get("selling_point_images", 0) or 0),
            }
        result_summary = {
            "main_images": main_count,
            "model_images": model_count,
            "selling_point_images": selling_point_count,
            "transparent_images": transparent_count,
            "vip_images": vip_count,
            "render_workers": worker_count,
            "local_model_used": int(local_model_assistant is not None),
        }
        _write_manifest(
            output_root,
            generated_paths,
            manifest_parameters,
            source_mappings,
            source_signatures,
            result_summary,
            serialized_source_records,
        )
    result: dict[str, object] = {
        "sources": len(sources),
        "main_images": main_count,
        "model_images": model_count,
        "selling_point_images": selling_point_count,
        "ai_assist": ai_provider is not None,
        "ai_model": ai_provider.model if ai_provider is not None else None,
        "ai_errors": list(ai_provider.errors) if ai_provider is not None else [],
        "transparent_images": transparent_count,
        "vip_images": vip_count,
        "render_workers": worker_count,
        "local_model_used": local_model_assistant is not None,
        "local_model_errors": local_model_errors,
        "cancelled": cancelled,
        "retried_only": source_filter is not None,
        "incremental_reused": len(reusable_records) if source_filter is None else 0,
        "stale_outputs_removed": stale_removed,
        "failures": failures,
        "validation": {
            "passed": len(generated_paths) - len({issue.path for issue in validation_issues if issue.level == "failed"}),
            "warnings": [
                {"code": issue.code, "path": str(issue.path), "message": issue.message}
                for issue in validation_issues
                if issue.level == "warning"
            ],
            "failed": [
                {"code": issue.code, "path": str(issue.path), "message": issue.message}
                for issue in validation_issues
                if issue.level == "failed"
            ],
        },
    }
    if category_decision is not None:
        result.update(
            {
                "category": category_decision.category,
                "category_source": category_decision.source,
                "category_confidence": category_decision.confidence,
                "composition_templates": {
                    size_name: {
                        "scale": template_manager.get(category_decision.category, size_name).scale,
                        "position_x": template_manager.get(category_decision.category, size_name).position_x,
                        "position_y": template_manager.get(category_decision.category, size_name).position_y,
                    }
                    for size_name in selected_sizes
                },
            }
        )
    if enable_material_understanding:
        result.update(
            {
                "material_understanding": True,
                "main_product_sources": sum(
                    decision.source_type == "main_product"
                    for decision in source_type_decisions.values()
                ),
                "detail_sources": sum(
                    decision.source_type == "detail"
                    for decision in source_type_decisions.values()
                ),
                "model_sources": sum(
                    decision.source_type == "model"
                    for decision in source_type_decisions.values()
                ),
                "source_types": {
                    source.relative_to(product_root).as_posix(): {
                        "type": decision.source_type,
                        "confidence": decision.confidence,
                        "source": decision.source,
                        "reason": decision.reason,
                        "metadata": decision.metadata or {},
                    }
                    for source, decision in source_type_decisions.items()
                },
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product-root", type=Path, default=PRODUCT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    result = generate_images(args.product_root, args.output_root)
    print(
        f"Generated {result['main_images']} main images, "
        f"{result['transparent_images']} transparent product images and "
        f"{result['vip_images']} VIP square images."
    )


if __name__ == "__main__":
    main()
