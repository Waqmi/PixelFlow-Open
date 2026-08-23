"""Load and validate category composition templates from JSON."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .category_classifier import CATEGORY_IDS


@dataclass(frozen=True)
class CategoryTemplate:
    category: str
    scale: float
    position_x: float
    position_y: float
    size_name: str | None = None


class CategoryTemplateManager:
    def __init__(self, template_path: Path) -> None:
        self.template_path = template_path
        self.templates = self._load()

    def _load(self) -> dict[str, CategoryTemplate]:
        if not self.template_path.exists():
            raise FileNotFoundError(f"品类模板文件不存在：{self.template_path}")
        payload = json.loads(self.template_path.read_text(encoding="utf-8"))
        templates: dict[str, CategoryTemplate] = {}
        for category in CATEGORY_IDS:
            raw = payload.get(category)
            if not isinstance(raw, dict):
                raise ValueError(f"品类模板缺少配置：{category}")
            if "scale" in raw:
                templates[category] = self._make_template(category, raw)
            else:
                sizes = raw.get("sizes")
                if not isinstance(sizes, dict):
                    raise ValueError(f"品类模板缺少 sizes 配置：{category}")
                for size_name, size_raw in sizes.items():
                    if not isinstance(size_raw, dict):
                        raise ValueError(f"品类模板尺寸配置无效：{category}/{size_name}")
                    templates[f"{category}:{size_name}"] = self._make_template(
                        category, size_raw, size_name
                    )
        return templates

    @staticmethod
    def _make_template(
        category: str, raw: dict[str, object], size_name: str | None = None
    ) -> CategoryTemplate:
        scale = float(raw["scale"])
        position_x = float(raw.get("position_x", 0.5))
        position_y = float(raw.get("position_y", 0.5))
        if not 0 < scale <= 1:
            raise ValueError(f"品类模板 scale 无效：{category}")
        if not 0 <= position_x <= 1 or not 0 <= position_y <= 1:
            raise ValueError(f"品类模板 position 无效：{category}")
        return CategoryTemplate(category, scale, position_x, position_y, size_name)

    def get(self, category: str, size_name: str | None = None) -> CategoryTemplate:
        if size_name is not None:
            sized = self.templates.get(f"{category}:{size_name}")
            if sized is not None:
                return sized
        try:
            return self.templates[category]
        except KeyError as error:
            raise ValueError(f"没有对应的品类模板：{category}") from error
