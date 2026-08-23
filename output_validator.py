"""Non-destructive validation for PixelFlow outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    code: str
    path: Path
    message: str


def validate_output(
    path: Path,
    expected_size: tuple[int, int],
    expected_mode: str,
    max_size_mb: float | None = None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    try:
        with Image.open(path) as image:
            image.load()
            if image.size != expected_size:
                issues.append(
                    ValidationIssue("failed", "invalid_dimensions", path, "输出像素尺寸不正确")
                )
            if image.mode != expected_mode:
                issues.append(
                    ValidationIssue("failed", "invalid_mode", path, f"输出模式应为 {expected_mode}")
                )
            if expected_mode == "RGBA":
                corners = (
                    image.getpixel((0, 0))[3],
                    image.getpixel((image.width - 1, 0))[3],
                    image.getpixel((0, image.height - 1))[3],
                    image.getpixel((image.width - 1, image.height - 1))[3],
                )
                if any(alpha != 0 for alpha in corners):
                    issues.append(
                        ValidationIssue("warning", "opaque_transparent_corner", path, "透明图四角不是完全透明")
                    )
    except Exception as error:  # noqa: BLE001 - validation must report corrupt output
        issues.append(ValidationIssue("failed", "invalid_image", path, f"无法读取输出：{error}"))
        return issues

    if max_size_mb is not None and path.stat().st_size > max_size_mb * 1024 * 1024:
        issues.append(ValidationIssue("warning", "file_size_exceeded", path, "文件大小超过设定上限，已保留目标像素尺寸"))
    return issues
