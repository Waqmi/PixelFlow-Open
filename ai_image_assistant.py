"""Optional OpenAI vision assistance for PixelFlow Beta.

The assistant only returns structured judgments. It never creates or edits the
final product image. Rendering remains local and deterministic in Pillow.
"""

from __future__ import annotations

import base64
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

from category_engine.source_type_classifier import SourceTypeDecision


API_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-5.6"
MAX_IMAGE_EDGE = 1280


class VisionAssistantError(RuntimeError):
    """A recoverable API or response-format error."""


def _clamp(value: object, lower: float, upper: float, default: float) -> float:
    try:
        return max(lower, min(upper, float(value)))
    except (TypeError, ValueError):
        return default


def _image_data_url(path: Path) -> str:
    """Create a bounded JPEG data URL so the desktop app sends small previews."""
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
        stream = BytesIO()
        image.save(stream, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _extract_output_text(payload: dict[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    chunks: list[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    if chunks:
        return "\n".join(chunks).strip()
    raise VisionAssistantError("API 没有返回可解析的文本结果")


SOURCE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "source_type": {
            "type": "string",
            "enum": ["main_product", "detail", "model"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
        "composition_mode": {
            "type": "string",
            "enum": ["safe_fit", "zoom_focus"],
        },
        "scale_adjustment_percent": {
            "type": "number",
            "minimum": -10,
            "maximum": 25,
        },
        "position_x": {"type": "number", "minimum": 0, "maximum": 1},
        "position_y": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "source_type",
        "confidence",
        "reason",
        "composition_mode",
        "scale_adjustment_percent",
        "position_x",
        "position_y",
    ],
}


class OpenAIVisionAssistant:
    """Small dependency-free Responses API adapter used by the Beta build."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout: float = 90.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OpenAI API Key 不能为空")
        self.api_key = api_key.strip()
        self.model = model.strip() or DEFAULT_MODEL
        self.timeout = timeout
        self.errors: list[str] = []

    @classmethod
    def from_environment(
        cls,
        api_key: str | None = None,
        model: str | None = None,
    ) -> "OpenAIVisionAssistant | None":
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get(
            "PIXELFLOW_OPENAI_API_KEY"
        )
        if not resolved_key:
            return None
        return cls(
            resolved_key,
            model or os.environ.get("PIXELFLOW_OPENAI_MODEL", DEFAULT_MODEL),
        )

    def _request_json(
        self,
        prompt: str,
        image_paths: list[Path],
        schema_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for path in image_paths:
            content.append(
                {
                    "type": "input_image",
                    "image_url": _image_data_url(path),
                    "detail": "high",
                }
            )
        payload = {
            "model": self.model,
            "input": [{"role": "user", "content": content}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        request = Request(
            API_URL,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")[:500]
            raise VisionAssistantError(f"API 请求失败（HTTP {error.code}）：{body}") from error
        except (URLError, TimeoutError, OSError) as error:
            raise VisionAssistantError(f"无法连接 API：{error}") from error
        except json.JSONDecodeError as error:
            raise VisionAssistantError("API 返回内容不是有效 JSON") from error
        text = _extract_output_text(response_payload)
        try:
            result = json.loads(text)
        except json.JSONDecodeError as error:
            raise VisionAssistantError("API 返回的结构化内容无法解析") from error
        if not isinstance(result, dict):
            raise VisionAssistantError("API 返回的结构化结果不是对象")
        return result

    def classify(
        self,
        source: Path,
        siblings: tuple[Path, ...] | list[Path] = (),
        context: dict[str, Any] | None = None,
    ) -> SourceTypeDecision | None:
        context = context or {}
        category = str(context.get("category") or "未指定")
        template_scale = context.get("template_scale")
        prompt = f"""你是电商主图素材审核助手。只判断现有图片，不生成图片。

将图片分类为：main_product（商品白底/完整商品图）、detail（卖点细节图）、model（含人物穿着或上身展示的模特图）。
当前用户选择的商品品类模板是：{category}。
当前方图模板主体比例参考值是：{template_scale if template_scale is not None else '未知'}。

对于 model：给出是否适合在保持原背景的前提下适度放大构图。只有明显能突出商品区域且不应严重裁掉人物主体时，才使用 zoom_focus；否则使用 safe_fit。
scale_adjustment_percent 是相对当前安全构图的建议增量，通常为 0 到 15；position_x 和 position_y 是 0 到 1 的画布归一化位置。
不要虚构图片中看不到的卖点，不要建议重绘或替换商品。
"""
        try:
            result = self._request_json(prompt, [source], "pixel_flow_source_judgment", SOURCE_SCHEMA)
            source_type = result.get("source_type")
            if source_type not in {"main_product", "detail", "model"}:
                raise VisionAssistantError("API 返回了不支持的素材类型")
            confidence = _clamp(result.get("confidence"), 0, 1, 0.0)
            scale_adjustment = _clamp(
                result.get("scale_adjustment_percent"), -10, 25, 0.0
            )
            metadata = {
                "composition_mode": result.get("composition_mode", "safe_fit"),
                "scale_adjustment_percent": scale_adjustment,
                "position_x": _clamp(result.get("position_x"), 0, 1, 0.5),
                "position_y": _clamp(result.get("position_y"), 0, 1, 0.5),
                "category": category,
            }
            return SourceTypeDecision(
                source_type,
                confidence,
                "openai_vision_beta",
                str(result.get("reason") or "API 判断完成"),
                metadata,
            )
        except (VisionAssistantError, OSError, ValueError) as error:
            self.errors.append(f"{source.name}：{error}")
            return None

    def match_selling_point(
        self,
        source: Path,
        selling_points: tuple[str, ...] | list[str],
    ) -> str | None:
        points = tuple(dict.fromkeys(point.strip() for point in selling_points if point.strip()))
        if not points:
            return None
        enum_points = list(points) + ["无匹配"]
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "selling_point": {"type": "string", "enum": enum_points},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
            },
            "required": ["selling_point", "confidence", "reason"],
        }
        prompt = """你是电商细节图与卖点文案匹配助手。只根据图片中可见的结构、材质和使用特征判断，不生成或修改图片。
请从给定卖点中选择最能被当前细节图直接支持的一个；如果图片无法可靠支持任何卖点，选择“无匹配”。不要因为文案听起来合理就强行匹配。

候选卖点：
""" + "\n".join(f"- {point}" for point in points)
        try:
            result = self._request_json(prompt, [source], "pixel_flow_selling_point_match", schema)
            matched = result.get("selling_point")
            if matched == "无匹配" or matched not in points:
                return None
            return str(matched)
        except (VisionAssistantError, OSError, ValueError) as error:
            self.errors.append(f"{source.name}：卖点匹配失败：{error}")
            return None
