"""Rule-first source material classification with an optional AI boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable, Protocol, Sequence

from PIL import Image

SOURCE_TYPE_IDS = ("main_product", "detail", "model", "model_detail")

MODEL_KEYWORDS = (
    "model", "look", "lookbook", "onbody", "outfit", "wear",
    "模特", "上身", "上脚", "穿搭", "搭配", "实拍穿着", "街拍", "人像", "人物",
    "双人", "多人", "合照", "合影", "情侣", "团体",
)


@dataclass(frozen=True)
class SourceTypeDecision:
    source_type: str
    confidence: float
    source: str
    reason: str
    metadata: dict[str, object] | None = None


class AISourceTypeProvider(Protocol):
    """Future adapter for a local or remote image understanding provider."""

    def classify(
        self,
        source: Path,
        siblings: Sequence[Path],
        context: dict[str, object] | None = None,
    ) -> SourceTypeDecision | None:
        ...


def _normalized_stem(source: Path) -> str:
    stem = source.stem.lower().strip()
    for suffix in ("_jpg", "_jpeg", "_png", " jpg", " jpeg", " png"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem.rstrip(" _-.")


def _has_transparency(image: Image.Image) -> bool:
    if "A" not in image.getbands() and "transparency" not in image.info:
        return False
    return image.convert("RGBA").getchannel("A").getextrema()[0] < 255


def _edge_background(image: Image.Image) -> tuple[tuple[int, int, int], bool]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    step = max(1, max(width, height) // 128)
    edge = []
    for x in range(0, width, step):
        edge.extend((rgb.getpixel((x, 0)), rgb.getpixel((x, height - 1))))
    for y in range(0, height, step):
        edge.extend((rgb.getpixel((0, y)), rgb.getpixel((width - 1, y))))
    background = tuple(int(median(channel)) for channel in zip(*edge))
    uniform = sum(
        max(abs(channel - base) for channel, base in zip(pixel, background)) <= 24
        for pixel in edge
    ) / max(1, len(edge)) >= 0.88
    return background, uniform


def _foreground_metrics(image: Image.Image, background: tuple[int, int, int]) -> tuple[float, float, bool]:
    rgb = image.convert("RGB")
    scale = min(1.0, 512 / max(rgb.width, rgb.height))
    preview = rgb.resize(
        (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))),
        Image.Resampling.BILINEAR,
    )
    width, height = preview.size
    pixels = list(preview.getdata())
    foreground = [
        index
        for index, pixel in enumerate(pixels)
        if max(abs(channel - base) for channel, base in zip(pixel, background)) > 32
    ]
    if not foreground:
        return 0.0, 0.0, False
    left = min(index % width for index in foreground)
    top = min(index // width for index in foreground)
    right = max(index % width for index in foreground) + 1
    bottom = max(index // width for index in foreground) + 1
    width_ratio = (right - left) / width
    height_ratio = (bottom - top) / height
    touches_edge = left <= max(1, round(width * 0.015)) or top <= max(1, round(height * 0.015))
    touches_edge = touches_edge or right >= width - max(1, round(width * 0.015))
    touches_edge = touches_edge or bottom >= height - max(1, round(height * 0.015))
    return width_ratio, height_ratio, touches_edge


class RuleSourceTypeClassifier:
    """Classify complete product images before category composition."""

    def __init__(
        self,
        ai_provider: AISourceTypeProvider | None = None,
        prefer_ai: bool = False,
        ai_context: dict[str, object] | None = None,
    ) -> None:
        self.ai_provider = ai_provider
        self.prefer_ai = prefer_ai
        self.ai_context = ai_context or {}

    @staticmethod
    def _is_override(source: Path, overrides: dict[str, str]) -> str | None:
        for key in (str(source), source.name, source.as_posix()):
            value = overrides.get(key)
            if value:
                return value
        return None

    def classify(
        self,
        source: Path,
        siblings: Sequence[Path] = (),
        manual_type: str | None = None,
    ) -> SourceTypeDecision:
        if manual_type is not None:
            if manual_type not in SOURCE_TYPE_IDS:
                raise ValueError(f"不支持的素材类型：{manual_type}")
            return SourceTypeDecision(manual_type, 1.0, "manual", "手动指定")

        if self.ai_provider is not None and self.prefer_ai:
            try:
                decision = self.ai_provider.classify(source, siblings, self.ai_context)
            except TypeError:
                # Preserve compatibility with older local providers.
                decision = self.ai_provider.classify(source, siblings)
            if decision is not None and decision.source_type in SOURCE_TYPE_IDS:
                return decision

        name_context = f"{source.parent.name} {source.stem}".lower()
        if any(keyword in name_context for keyword in MODEL_KEYWORDS):
            return SourceTypeDecision("model", 0.96, "rules", "文件夹或文件名包含模特素材标记")

        with Image.open(source) as image:
            if _has_transparency(image):
                return SourceTypeDecision("main_product", 0.99, "rules", "PNG 含透明通道")
            background, uniform = _edge_background(image)
            width_ratio, height_ratio, touches_edge = _foreground_metrics(image, background)

        normalized = _normalized_stem(source)
        paired = any(
            sibling != source
            and sibling.parent == source.parent
            and _normalized_stem(sibling) == normalized
            and sibling.suffix.lower() != source.suffix.lower()
            for sibling in siblings
        )
        if paired:
            return SourceTypeDecision("main_product", 0.98, "rules", "同名 JPG/PNG 素材组合")

        background_is_light = max(background) - min(background) <= 28 and sum(background) / 3 >= 205
        if uniform and background_is_light and not touches_edge:
            return SourceTypeDecision(
                "main_product",
                0.90 if width_ratio < 0.90 and height_ratio < 0.90 else 0.76,
                "rules",
                "浅色均匀背景且主体未贴边",
            )
        if touches_edge:
            return SourceTypeDecision("detail", 0.88, "rules", "主体或画面内容贴近边缘")

        if uniform and background_is_light:
            return SourceTypeDecision("main_product", 0.68, "rules", "浅色背景，主体轮廓不明确")

        if self.ai_provider is not None:
            try:
                decision = self.ai_provider.classify(source, siblings, self.ai_context)
            except TypeError:
                decision = self.ai_provider.classify(source, siblings)
            if decision is not None and decision.source_type in SOURCE_TYPE_IDS:
                return decision
        return SourceTypeDecision("detail", 0.55, "fallback", "规则无法确认，按细节图处理")

    def classify_many(
        self,
        sources: Sequence[Path],
        overrides: dict[str, str] | None = None,
        progress: Callable[[int, int, Path, SourceTypeDecision], None] | None = None,
    ) -> dict[Path, SourceTypeDecision]:
        overrides = overrides or {}
        decisions: dict[Path, SourceTypeDecision] = {}
        for index, source in enumerate(sources, start=1):
            try:
                decision = self.classify(
                    source,
                    sources,
                    manual_type=self._is_override(source, overrides),
                )
            except Exception as error:  # noqa: BLE001 - a bad file must not block confirmation
                decision = SourceTypeDecision("detail", 0.0, "error", f"无法读取素材：{error}")
            decisions[source] = decision
            if progress is not None:
                progress(index, len(sources), source, decision)
        return decisions
