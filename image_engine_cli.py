"""JSON Lines process boundary for the future Flutter desktop client."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

from image_engine_api import RESOURCE_ROOT, generate_images

PROTOCOL_VERSION = 1


def emit(event: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def fail(job_id: str | None, code: str, message: str) -> None:
    emit(
        {
            "version": PROTOCOL_VERSION,
            "type": "failed",
            "jobId": job_id,
            "error": {"code": code, "message": message},
        }
    )


def start_job(job_id: str, options: dict[str, Any]) -> None:
    product_root = Path(str(options.get("productRoot", ""))).expanduser()
    output_root = Path(str(options.get("outputRoot", ""))).expanduser()
    selected_sizes = tuple(str(value) for value in options.get("selectedSizes", ()))
    include_logo = bool(options.get("includeLogo", True))
    include_vip = bool(options.get("includeVip", True))
    raw_limit = options.get("maxSizeMb")
    max_size_mb = float(raw_limit) if raw_limit not in (None, "", 0, 0.0) else None
    category_override = options.get("category")
    enable_category_template = bool(options.get("enableCategoryTemplate", False))
    enable_material_understanding = bool(options.get("enableMaterialUnderstanding", False))
    raw_overrides = options.get("sourceTypeOverrides", {})
    if not isinstance(raw_overrides, dict):
        raise ValueError("sourceTypeOverrides 必须是对象")
    source_type_overrides = {str(key): str(value) for key, value in raw_overrides.items()}
    raw_adjustments = options.get("sourceScaleAdjustments", {})
    if not isinstance(raw_adjustments, dict):
        raise ValueError("sourceScaleAdjustments 必须是对象")
    source_scale_adjustments = {str(key): float(value) for key, value in raw_adjustments.items()}
    raw_folder_categories = options.get("folderCategoryOverrides", {})
    if not isinstance(raw_folder_categories, dict):
        raise ValueError("folderCategoryOverrides 必须是对象")
    folder_category_overrides = {str(key): str(value) for key, value in raw_folder_categories.items()}
    ai_assist = bool(options.get("aiAssist", False))
    ai_api_key = str(options.get("aiApiKey") or "") or None
    ai_model = str(options.get("aiModel") or "") or None

    emit({"version": PROTOCOL_VERSION, "type": "started", "jobId": job_id})

    def progress(completed: int, total: int, message: str) -> None:
        emit(
            {
                "version": PROTOCOL_VERSION,
                "type": "progress",
                "jobId": job_id,
                "completed": completed,
                "total": total,
                "message": message,
            }
        )

    result = generate_images(
        product_root,
        output_root,
        resource_root=RESOURCE_ROOT,
        selected_sizes=selected_sizes,
        include_logo=include_logo,
        include_vip=include_vip,
        max_size_mb=max_size_mb,
        category_override=str(category_override) if category_override else None,
        enable_category_template=enable_category_template,
        enable_material_understanding=enable_material_understanding,
        source_type_overrides=source_type_overrides,
        source_scale_adjustments=source_scale_adjustments,
        folder_category_overrides=folder_category_overrides,
        ai_assist=ai_assist,
        ai_api_key=ai_api_key,
        ai_model=ai_model,
        progress=progress,
    )
    emit(
        {
            "version": PROTOCOL_VERSION,
            "type": "completed",
            "jobId": job_id,
            "result": result,
        }
    )


def main() -> int:
    active_job_id: str | None = None
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if request.get("version") != PROTOCOL_VERSION:
                fail(request.get("jobId"), "unsupported_protocol", "不支持的 Engine 协议版本")
                continue
            request_type = request.get("type")
            if request_type == "start":
                if active_job_id is not None:
                    fail(request.get("jobId"), "busy", "Engine 当前已有任务在运行")
                    continue
                active_job_id = str(request.get("jobId") or "job")
                try:
                    start_job(active_job_id, request.get("options") or {})
                finally:
                    active_job_id = None
            elif request_type == "ping":
                emit({"version": PROTOCOL_VERSION, "type": "pong"})
            elif request_type == "cancel":
                # Cancellation is reserved for the next engine step. The
                # current renderer is synchronous and has no safe interruption
                # point between file writes yet.
                fail(request.get("jobId"), "cancel_unavailable", "当前 Engine 版本尚不支持取消任务")
            else:
                fail(request.get("jobId"), "invalid_request", "未知的 Engine 请求类型")
        except json.JSONDecodeError as error:
            fail(None, "invalid_json", f"请求不是有效 JSON：{error.msg}")
        except Exception as error:  # noqa: BLE001 - protocol must report all job failures
            emit(
                {
                    "version": PROTOCOL_VERSION,
                    "type": "failed",
                    "jobId": active_job_id,
                    "error": {"code": "engine_error", "message": str(error)},
                }
            )
            traceback.print_exc(file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
