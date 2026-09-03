"""Optional local ONNX person detection and safe model-photo composition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class PersonDetection:
    """One person candidate in original-image pixel coordinates."""

    bbox: tuple[float, float, float, float]
    confidence: float
    keypoints: tuple[tuple[float, float, float], ...] = ()


BODY_KEYPOINT_CONFIDENCE = 0.28


def shirt_protected_region(
    detection: PersonDetection,
    image_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Estimate the head-to-hip safety region for a model shirt crop."""
    image_width, image_height = image_size
    x1, y1, x2, y2 = detection.bbox
    body_height = max(1.0, y2 - y1)
    visible = {
        index: point
        for index, point in enumerate(detection.keypoints)
        if point[2] >= BODY_KEYPOINT_CONFIDENCE
    }
    upper_x = [
        visible[index][0]
        for index in (0, 1, 2, 3, 4, 5, 6, 11, 12)
        if index in visible
    ]
    shoulder_width = (
        abs(visible[5][0] - visible[6][0])
        if 5 in visible and 6 in visible
        else max(1.0, (x2 - x1) * 0.45)
    )
    horizontal_margin = max(body_height * 0.03, shoulder_width * 0.18)
    left = min(upper_x or [x1]) - horizontal_margin
    right = max(upper_x or [x2]) + horizontal_margin
    face_y = [visible[index][1] for index in range(5) if index in visible]
    hip_y = [visible[index][1] for index in (11, 12) if index in visible]
    top = min([y1, *face_y]) - body_height * 0.02
    bottom = max(hip_y or [y1 + body_height * 0.58]) + body_height * 0.08
    return (
        max(0.0, left),
        max(0.0, top),
        min(float(image_width), right),
        min(float(image_height), bottom),
    )


def body_visibility(
    detections: list[PersonDetection],
    image_size: tuple[int, int] | None = None,
) -> dict[str, object]:
    """Summarize visible body regions without tying the result to a category.

    The pose model can see partial people (for example, a lower-body outfit
    photo).  Keeping this evidence separate from ``model_view`` lets the UI
    make category-compatibility decisions without turning every partial shot
    into a generic detail image.
    """
    visible = [
        index
        for detection in detections
        for index, point in enumerate(detection.keypoints)
        if point[2] >= BODY_KEYPOINT_CONFIDENCE
    ]
    visible_set = set(visible)
    head = bool(visible_set.intersection(range(0, 5)))
    upper_points = visible_set.intersection(range(5, 11))
    lower_points = visible_set.intersection(range(11, 15))
    feet_points = visible_set.intersection((15, 16))
    upper = head or len(upper_points) >= 2
    lower = len(lower_points) >= 2 or bool(feet_points)
    # Keep group-specific evidence separate from the broad upper/lower flags.
    # A shirt group requires every detected person's head to remain visible,
    # while pants/shorts/shoes may be composed without heads.
    head_visible_count = sum(
        any(
            len(detection.keypoints) > index
            and detection.keypoints[index][2] >= BODY_KEYPOINT_CONFIDENCE
            for index in range(0, 5)
        )
        for detection in detections
    )
    top_clip_threshold = max(1.0, (image_size[1] * 0.01 if image_size else 1.0))
    head_clipped = any(
        detection.bbox[1] <= top_clip_threshold
        or any(
            len(detection.keypoints) > index
            and detection.keypoints[index][2] >= BODY_KEYPOINT_CONFIDENCE
            and detection.keypoints[index][1] <= top_clip_threshold
            for index in range(0, 5)
        )
        for detection in detections
    )
    return {
        "head": head,
        "upper_body": upper,
        "lower_body": lower,
        "feet": bool(feet_points),
        "torso_or_hip": bool(upper_points.intersection((5, 6))) or bool(lower_points.intersection((11, 12))),
        "visible_keypoints": len(visible_set),
        "person_count": len(detections),
        "head_visible_count": head_visible_count,
        "all_heads_visible": bool(detections)
        and head_visible_count == len(detections)
        and not head_clipped,
    }


def category_compatibility(
    category: str | None,
    visibility: dict[str, object] | None,
) -> tuple[str, str]:
    """Return ``compatible``, ``incompatible`` or ``unknown`` for a model shot.

    Only clear mismatches are marked incompatible.  Broad categories such as
    bags and accessories remain manual-review cases instead of being silently
    skipped based on a weak pose signal.
    """
    if not category or not visibility:
        return "unknown", "缺少品类或人体区域信息"
    head = bool(visibility.get("head"))
    upper = bool(visibility.get("upper_body"))
    lower = bool(visibility.get("lower_body"))
    feet = bool(visibility.get("feet"))
    if category in {"shoes", "socks"}:
        if feet and bool(visibility.get("feet_clipped")):
            return "incompatible", "脚部贴近画面边缘，鞋部构图不完整"
        return ("compatible", "检测到脚部") if feet else ("incompatible", "未检测到脚部")
    if category in {"pants", "long_pants", "shorts"}:
        return ("compatible", "检测到下半身") if lower else ("incompatible", "未检测到下半身")
    if category == "shirt":
        if int(visibility.get("person_count", 0) or 0) >= 2 and not bool(
            visibility.get("all_heads_visible")
        ):
            return "incompatible", "双人上衣要求两个人头部完整可见"
        return ("compatible", "检测到上半身") if upper else ("incompatible", "未检测到上半身")
    if category == "hat":
        return ("compatible", "检测到头部") if head else ("incompatible", "未检测到头部")
    return "unknown", "该品类需要人工确认适配性"


def credible_model_composition(composition: dict[str, object] | None) -> bool:
    """Reject pose hallucinations on close-up/detail images.

    A single-person pose is only credible when it contains enough landmarks
    and occupies a meaningful portion of the source frame. Group detections
    remain eligible because two independent people are a strong signal.
    """
    if not isinstance(composition, dict):
        return False
    if composition.get("model_view") == "group":
        return int(composition.get("person_count", 0) or 0) >= 2
    view = composition.get("model_view")
    if view not in {"full_body", "upper_body", "lower_body"}:
        return False
    visibility = composition.get("body_visibility")
    if not isinstance(visibility, dict):
        return False
    return (
        int(composition.get("person_count", 0) or 0) == 1
        and int(visibility.get("visible_keypoints", 0) or 0) >= 6
        and float(visibility.get("person_box_height_ratio", 0.0) or 0.0) >= 0.35
    )


def find_onnx_model(model_dir: str | Path) -> Path | None:
    path = Path(model_dir).expanduser()
    if not path.is_dir():
        return None
    models = sorted(path.glob("*.onnx"))
    return models[0] if models else None


def find_compatible_onnx_model(
    model_dir: str | Path,
    fallback_model_dir: str | Path | None = None,
) -> Path | None:
    """Return the first loadable model, optionally using a bundled fallback."""
    candidates: list[Path] = []
    for directory in (model_dir, fallback_model_dir):
        if not directory:
            continue
        path = Path(directory).expanduser()
        if not path.is_dir():
            continue
        for candidate in sorted(path.glob("*.onnx")):
            if candidate not in candidates:
                candidates.append(candidate)
    if not candidates:
        return None
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    for candidate in candidates:
        try:
            ort.InferenceSession(str(candidate), providers=["CPUExecutionProvider"])
        except Exception:  # noqa: BLE001 - try the next compatible candidate
            continue
        return candidate
    return None


class LocalModelAssistant:
    """Run a YOLO-style ONNX model when the optional runtime is available.

    The parser accepts both a one-class person pose model (for example
    YOLOv8n-pose exported to ONNX) and a COCO pose/detection export. A model
    failure is intentionally isolated so the caller can keep rule fallback.
    """

    def __init__(self, model_path: str | Path, confidence_threshold: float = 0.28) -> None:
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as error:  # pragma: no cover - depends on packaged extras
            raise RuntimeError("未安装 ONNX Runtime 或 NumPy") from error

        self._np = np
        self.model_path = Path(model_path)
        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        shape = input_meta.shape
        self.input_height = self._static_dimension(shape[2] if len(shape) > 2 else None, 640)
        self.input_width = self._static_dimension(shape[3] if len(shape) > 3 else None, 640)
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def _static_dimension(value: Any, fallback: int) -> int:
        return int(value) if isinstance(value, int) and value > 0 else fallback

    @staticmethod
    def inspect(
        model_dir: str | Path,
        fallback_model_dir: str | Path | None = None,
    ) -> tuple[bool, str]:
        model_path = find_compatible_onnx_model(model_dir, fallback_model_dir)
        if model_path is None:
            return False, "未检测到可加载的 ONNX 模型文件"
        try:
            LocalModelAssistant(model_path)
        except Exception as error:  # noqa: BLE001 - surface a friendly settings status
            return False, f"模型无法加载：{error}"
        return True, f"模型可用：{model_path.name}"

    def _letterbox(self, image: Image.Image):
        np = self._np
        rgb = image.convert("RGB")
        scale = min(self.input_width / rgb.width, self.input_height / rgb.height)
        resized_size = (
            max(1, round(rgb.width * scale)),
            max(1, round(rgb.height * scale)),
        )
        resized = rgb.resize(resized_size, Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.input_width, self.input_height), (114, 114, 114))
        pad_x = (self.input_width - resized.width) / 2
        pad_y = (self.input_height - resized.height) / 2
        canvas.paste(resized, (round(pad_x), round(pad_y)))
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))[None, ...]
        return array, scale, pad_x, pad_y

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            return 1.0 / (1.0 + pow(2.718281828, -value))
        exponent = pow(2.718281828, value)
        return exponent / (1.0 + exponent)

    def _decode(self, output, image_size: tuple[int, int], scale: float, pad_x: float, pad_y: float) -> list[PersonDetection]:
        np = self._np
        data = np.asarray(output)
        while data.ndim > 2:
            data = data[0]
        if data.ndim != 2:
            return []
        if data.shape[0] < data.shape[1] and data.shape[0] <= 200:
            data = data.T
        rows, channels = data.shape
        keypoint_size = 51 if channels >= 56 else 0
        class_start = 4
        class_end = channels - keypoint_size
        if class_end <= class_start:
            return []
        detections: list[PersonDetection] = []
        source_width, source_height = image_size
        for row in data:
            class_values = row[class_start:class_end]
            if len(class_values) == 0:
                continue
            # COCO exports use class 0 for person; one-class pose exports have
            # one score. Do not let a non-person COCO class become a model photo.
            score = float(class_values[0])
            if score > 1.0:
                score = self._sigmoid(score)
            if score < self.confidence_threshold:
                continue
            cx, cy, width, height = (float(value) for value in row[:4])
            if max(abs(cx), abs(cy), abs(width), abs(height)) <= 2.0:
                cx *= self.input_width
                cy *= self.input_height
                width *= self.input_width
                height *= self.input_height
            x1 = max(0.0, min(source_width, (cx - width / 2 - pad_x) / scale))
            y1 = max(0.0, min(source_height, (cy - height / 2 - pad_y) / scale))
            x2 = max(0.0, min(source_width, (cx + width / 2 - pad_x) / scale))
            y2 = max(0.0, min(source_height, (cy + height / 2 - pad_y) / scale))
            if x2 <= x1 or y2 <= y1:
                continue
            keypoints: tuple[tuple[float, float, float], ...] = ()
            if keypoint_size:
                values = row[-keypoint_size:].reshape(-1, 3)
                keypoint_list = []
                for point_x, point_y, point_conf in values:
                    confidence = float(point_conf)
                    if confidence > 1.0:
                        confidence = self._sigmoid(confidence)
                    keypoint_list.append(
                        (
                            max(0.0, min(source_width, (float(point_x) - pad_x) / scale)),
                            max(0.0, min(source_height, (float(point_y) - pad_y) / scale)),
                            confidence,
                        )
                    )
                keypoints = tuple(keypoint_list)
            detections.append(PersonDetection((x1, y1, x2, y2), score, keypoints))
        return self._nms(detections)

    @staticmethod
    def _nms(detections: list[PersonDetection], iou_threshold: float = 0.55) -> list[PersonDetection]:
        remaining = sorted(detections, key=lambda item: item.confidence, reverse=True)
        selected: list[PersonDetection] = []
        while remaining:
            current = remaining.pop(0)
            selected.append(current)
            cx1, cy1, cx2, cy2 = current.bbox
            current_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
            filtered = []
            for candidate in remaining:
                x1 = max(cx1, candidate.bbox[0])
                y1 = max(cy1, candidate.bbox[1])
                x2 = min(cx2, candidate.bbox[2])
                y2 = min(cy2, candidate.bbox[3])
                intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                candidate_area = max(0.0, candidate.bbox[2] - candidate.bbox[0]) * max(0.0, candidate.bbox[3] - candidate.bbox[1])
                union = current_area + candidate_area - intersection
                if union <= 0 or intersection / union < iou_threshold:
                    filtered.append(candidate)
            remaining = filtered
        return selected

    def detect(self, source: Path) -> list[PersonDetection]:
        with Image.open(source) as opened:
            original_size = opened.size
            # The pose model always letterboxes to a small fixed input.  Ask
            # JPEG decoders for a reduced native decode first; this saves most
            # of the preparation time for camera-sized source photos without
            # changing the original image used for final rendering.
            opened.draft("RGB", (self.input_width, self.input_height))
            image = opened.convert("RGB")
        input_array, scale, pad_x, pad_y = self._letterbox(image)
        outputs = self.session.run(None, {self.input_name: input_array})
        if not outputs:
            return []
        detections = self._decode(outputs[0], image.size, scale, pad_x, pad_y)
        if image.size == original_size:
            return detections
        scale_x = original_size[0] / image.width
        scale_y = original_size[1] / image.height
        return [
            PersonDetection(
                tuple(
                    value * (scale_x if index % 2 == 0 else scale_y)
                    for index, value in enumerate(detection.bbox)
                ),
                detection.confidence,
                tuple(
                    (point_x * scale_x, point_y * scale_y, confidence)
                    for point_x, point_y, confidence in detection.keypoints
                ),
            )
            for detection in detections
        ]

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, value))

    def _composition(
        self,
        detection: PersonDetection,
        image_size: tuple[int, int],
        target_size: tuple[int, int],
        model_view: str,
        product_category: str | None,
        protected_region: tuple[float, float, float, float] | None = None,
    ) -> dict[str, object]:
        image_width, image_height = image_size
        x1, y1, x2, y2 = detection.bbox
        keypoints = [point for point in detection.keypoints if point[2] >= BODY_KEYPOINT_CONFIDENCE]
        # For a group, the detector union is the stable framing box.  Merging
        # keypoints from different people can spuriously expand it to the
        # image edges and destroy the true group midpoint.
        if keypoints and model_view != "group":
            x1 = min(x1, min(point[0] for point in keypoints))
            y1 = min(y1, min(point[1] for point in keypoints))
            x2 = max(x2, max(point[0] for point in keypoints))
            y2 = max(y2, max(point[1] for point in keypoints))
        has_feet = len(detection.keypoints) >= 17 and any(
            detection.keypoints[index][2] >= BODY_KEYPOINT_CONFIDENCE for index in (15, 16)
        )
        has_head_or_upper_body = len(detection.keypoints) >= 11 and any(
            detection.keypoints[index][2] >= BODY_KEYPOINT_CONFIDENCE for index in range(0, 7)
        )
        # A lower-body detail often has ankles and feet but no head or upper
        # torso. Treat it as a partial composition and leave more breathing
        # room instead of fitting it like a complete person.
        partial_lower_body = has_feet and not has_head_or_upper_body
        partial_upper_body = not has_feet and has_head_or_upper_body
        # The source is always scaled proportionally.  A full-body model can
        # focus on the selected product area while an upper/lower-body source
        # keeps its existing close composition.
        scale_adjustment = 0.0
        target_crop_height_ratio: float | None = None
        if product_category == "shirt" and protected_region is None and model_view != "group":
            protected_region = shirt_protected_region(detection, image_size)
        body_center_x = ((x1 + x2) / 2) / image_width
        # Use the torso axis when pose landmarks are available.  The full
        # person box includes extended arms, so its midpoint can pull an
        # otherwise centred model too far to one side.
        torso_centres = []
        for left_index, right_index in ((5, 6), (11, 12)):
            if (
                len(detection.keypoints) > right_index
                and detection.keypoints[left_index][2] >= BODY_KEYPOINT_CONFIDENCE
                and detection.keypoints[right_index][2] >= BODY_KEYPOINT_CONFIDENCE
            ):
                torso_centres.append(
                    (detection.keypoints[left_index][0] + detection.keypoints[right_index][0]) / 2
                )
        # A group must be framed by the group bounds, not by whichever
        # person's torso/feet happened to produce the strongest landmark.
        # Using an individual landmark here shifts two-person photos toward
        # one side when the subjects have different detection confidence.
        if model_view == "group":
            body_center_x = ((x1 + x2) / 2) / image_width
        elif torso_centres:
            body_center_x = (sum(torso_centres) / len(torso_centres)) / image_width
        if model_view != "group" and product_category in {"shoes", "socks"}:
            foot_centres = [
                point[0]
                for index in (15, 16)
                if len(detection.keypoints) > index
                for point in [detection.keypoints[index]]
                if point[2] >= BODY_KEYPOINT_CONFIDENCE
            ]
            if foot_centres:
                # Shoe outputs should be centered on the actual feet, not on
                # the torso axis. This keeps one-foot and two-foot shots
                # aligned consistently across square and portrait canvases.
                body_center_x = sum(foot_centres) / len(foot_centres) / image_width
        body_center_y = ((y1 + y2) / 2) / image_height
        category_focus = "group" if model_view == "group" else None
        if partial_upper_body:
            # Square outputs should feature the jacket and face; portrait
            # outputs retain a little more of the lower garment.
            target_ratio = target_size[0] / max(1, target_size[1])
            body_center_y = 0.40 if target_ratio >= 0.9 else 0.46
        elif partial_lower_body:
            body_center_y = 0.56
        base_scale = max(target_size[0] / image_width, target_size[1] / image_height)
        if model_view == "upper_body":
            # Close portraits are often shot at slightly different distances.
            # Normalize the detected person height before positioning so the
            # same batch does not contain one visibly smaller half-body shot.
            source_person_height_ratio = max(0.01, (y2 - y1) / image_height)
            # Avoid over-zooming close/half-body sources on square outputs.
            # The previous 0.87 target made small detector boxes expand too
            # aggressively compared with the supplied shirt reference.
            desired_person_height_ratio = 0.80
            scale_adjustment = max(
                0.0,
                (desired_person_height_ratio / source_person_height_ratio - 1.0) * 100,
            )
            if product_category == "shirt":
                # For shirt portraits that include the waist/hip landmarks,
                # normalize the visible lower boundary as well.  This keeps
                # long portraits from ending at the knees while leaving an
                # already well-framed source unchanged.
                hip_points = [
                    detection.keypoints[index][1]
                    for index in (11, 12)
                    if len(detection.keypoints) > index
                    and detection.keypoints[index][2] >= BODY_KEYPOINT_CONFIDENCE
                ]
                if hip_points:
                    hip_ratio = sum(hip_points) / len(hip_points) / image_height
                    target_hip_ratio = 0.80
                    if hip_ratio > 0.05:
                        scale_adjustment = max(
                            scale_adjustment,
                            (target_hip_ratio / hip_ratio - 1.0) * 100,
                        )
            if product_category == "shirt" and protected_region is not None:
                target_ratio = target_size[0] / max(1, target_size[1])
                protected_height_ratio = (
                    protected_region[3] - protected_region[1]
                ) / image_height
                crop_ratio = max(
                    0.34,
                    min(0.72, protected_height_ratio + (0.04 if target_ratio >= 0.9 else 0.08)),
                )
                target_crop_height_ratio = crop_ratio
                desired_scale = target_size[1] / (image_height * crop_ratio)
                scale_adjustment = max(
                    scale_adjustment,
                    max(0.0, desired_scale / base_scale - 1.0) * 100,
                )
                body_center_y = ((protected_region[1] + protected_region[3]) / 2) / image_height
                category_focus = "head_safe"
        if model_view in {"full_body", "lower_body", "group"} and product_category in {
            "shirt", "pants", "long_pants", "shorts", "shoes", "socks", "hat",
        }:
            body_height = max(1.0, y2 - y1)
            if product_category == "shirt":
                # Use the detected head-to-hip region, not a fixed body
                # fraction. This survives different poses and source crops.
                if protected_region is not None:
                    body_center_y = ((protected_region[1] + protected_region[3]) / 2) / image_height
                else:
                    body_center_y = (y1 + body_height * 0.28) / image_height
                category_focus = "head_safe"
            elif product_category in {"pants", "long_pants", "shorts"}:
                hip_points = [
                    detection.keypoints[index][1]
                    for index in (11, 12)
                    if len(detection.keypoints) > index
                    and detection.keypoints[index][2] >= BODY_KEYPOINT_CONFIDENCE
                ]
                # Keep the waist and the selected lower garment in frame;
                # unlike shirt composition, this intentionally allows the
                # head to leave the square crop.
                hip_center = sum(hip_points) / len(hip_points) if hip_points else y1 + body_height * 0.55
                body_center_y = (hip_center + body_height * (0.10 if product_category == "shorts" else 0.26)) / image_height
                category_focus = "lower_body"
            elif product_category in {"shoes", "socks"}:
                foot_points = [
                    detection.keypoints[index][1]
                    for index in (15, 16)
                    if len(detection.keypoints) > index
                    and detection.keypoints[index][2] >= BODY_KEYPOINT_CONFIDENCE
                ]
                foot_center = sum(foot_points) / len(foot_points) if foot_points else y1 + body_height * 0.92
                # Shoes use a dynamic protected region.  The pose landmarks
                # identify the ankle, while the detection box supplies a
                # conservative lower boundary for the shoe/sole.
                if protected_region is None:
                    foot_left = min(
                        [detection.keypoints[index][0] for index in (15, 16)
                         if len(detection.keypoints) > index
                         and detection.keypoints[index][2] >= BODY_KEYPOINT_CONFIDENCE]
                        or [x1]
                    )
                    foot_right = max(
                        [detection.keypoints[index][0] for index in (15, 16)
                         if len(detection.keypoints) > index
                         and detection.keypoints[index][2] >= BODY_KEYPOINT_CONFIDENCE]
                        or [x2]
                    )
                    foot_width = max(body_height * 0.12, (x2 - x1) * 0.16)
                    ankle_top = min(foot_points) if foot_points else y1 + body_height * 0.78
                    protected_region = (
                        max(0.0, foot_left - foot_width),
                        max(0.0, ankle_top - body_height * 0.16),
                        min(image_width, foot_right + foot_width),
                        min(image_height, y2 + body_height * 0.035),
                    )
                body_center_y = ((protected_region[1] + protected_region[3]) / 2) / image_height
                category_focus = "feet"
            elif product_category == "hat":
                # A hat is a head-focused product, so use the same safe
                # headroom rule as an upper-body shirt composition.
                body_center_y = (y1 + body_height * 0.10) / image_height
                category_focus = "head_safe"
            else:
                # Bags and accessories are usually worn around the torso or
                # hip. Keep a neutral body crop but do not force the face
                # into frame when the selected template targets the product.
                body_center_y = (y1 + body_height * 0.52) / image_height
                category_focus = "body"
            # Keep the selected garment as the dominant subject.  A square
            # shirt main image needs a tighter crop than a portrait output;
            # otherwise a full-body source remains visibly wider/longer than
            # an upper-body source in the same batch.  Portrait output keeps
            # more context so a usable full-body source is not over-cropped.
            target_ratio = target_size[0] / max(1, target_size[1])
            if model_view == "group":
                if product_category == "shirt":
                    # Group shirts use the union of both people’s dynamic
                    # head-to-hip safety regions. Square and portrait differ
                    # only in breathing room, never in the protected content.
                    protected_height_ratio = (
                        (protected_region[3] - protected_region[1]) / image_height
                        if protected_region is not None
                        else 0.46
                    )
                    crop_ratio = max(
                        0.38,
                        min(
                            0.72,
                            protected_height_ratio
                            + (0.04 if target_ratio >= 0.9 else 0.08),
                        ),
                    )
                else:
                    crop_ratio = 0.78 if target_ratio >= 0.9 else 0.88
            elif category_focus == "feet":
                # Both formats focus on the complete shoe.  Portrait keeps
                # slightly more calf/air than square, but neither format
                # should fall back to a full-body composition.
                protected_height_ratio = (
                    (protected_region[3] - protected_region[1]) / image_height
                    if protected_region is not None
                    else 0.24
                )
                crop_ratio = max(
                    0.22,
                    min(0.55, protected_height_ratio + (0.08 if target_ratio >= 0.9 else 0.12)),
                )
            elif product_category == "shirt":
                # Both formats target the same head-to-hip semantic region;
                # portrait receives a little more breathing room than square.
                protected_height_ratio = (
                    (protected_region[3] - protected_region[1]) / image_height
                    if protected_region is not None
                    else 0.46
                )
                crop_ratio = max(
                    0.38,
                    min(0.72, protected_height_ratio + (0.04 if target_ratio >= 0.9 else 0.08)),
                )
            else:
                crop_ratio = 0.36 if target_ratio >= 0.9 else 0.48
            desired_crop_height = image_height * crop_ratio
            target_crop_height_ratio = crop_ratio
            desired_scale = target_size[1] / desired_crop_height
            scale_adjustment = max(0.0, desired_scale / base_scale - 1.0) * 100
        if model_view == "group" and abs(body_center_x - 0.5) > 0.01:
            # When the source already fits the target width exactly, there
            # is no horizontal crop room and a group that was photographed
            # off-centre stays off-centre.  Add only the minimum zoom needed
            # to create enough crop room to move the group midpoint to the
            # canvas midpoint; this preserves both people as much as possible.
            base_fitted_width = image_width * base_scale
            required_fitted_width = target_size[0] / (2.0 * min(body_center_x, 1.0 - body_center_x))
            if required_fitted_width > base_fitted_width:
                scale_adjustment = max(
                    scale_adjustment,
                    (required_fitted_width / base_fitted_width - 1.0) * 100.0,
                )
        fitted_scale = base_scale * (1.0 + scale_adjustment / 100.0)
        fitted_width = image_width * fitted_scale
        fitted_height = image_height * fitted_scale

        def focus_position(center: float, fitted: float, target: int) -> float:
            center_px = center * fitted
            if fitted >= target:
                excess = fitted - target
                return self._clamp((center_px - target / 2) / excess) if excess else 0.5
            padding = target - fitted
            return self._clamp((target / 2 - center_px) / padding) if padding else 0.5

        # A facial keypoint does not include the hairline, so reserve 2% of
        # the original height above the detected person.  This is a hard
        # crop limit: a zoomed portrait may show less lower body, but must
        # never cut through the top of a visible head.
        face_points = [
            point
            for point in detection.keypoints[:5]
            if point[2] >= BODY_KEYPOINT_CONFIDENCE
        ]
        head_top = min([y1, *(point[1] for point in face_points)])
        safe_crop_top = max(0.0, head_top - image_height * 0.02) * fitted_scale
        vertical_excess = max(0.0, fitted_height - target_size[1])
        max_position_y = (
            self._clamp(safe_crop_top / vertical_excess)
            if vertical_excess > 0
            else 0.5
        )
        desired_head_margin = target_size[1] * 0.04
        head_position_y = (
            self._clamp((head_top * fitted_scale - desired_head_margin) / vertical_excess)
            if vertical_excess > 0
            else 0.5
        )
        # Keep a stable small margin above the head across source framings;
        # the safety limit still wins if the source cannot provide it.
        position_y = (
            min(head_position_y, max_position_y)
            if category_focus in {None, "head_safe"}
            else focus_position(body_center_y, fitted_height, target_size[1])
        )
        if category_focus == "feet":
            # Lower-body source photos commonly place the shoe at the bottom
            # edge even when the ankle landmark is higher. Bottom-align the
            # crop whenever the detected person reaches that edge so the toe
            # and sole are not cut off by centring on the ankle point.
            position_y = 1.0 if y2 / max(1, image_height) >= 0.85 else max(position_y, 0.82)

        composition = {
            "composition_mode": "zoom_focus",
            "scale_adjustment_percent": round(scale_adjustment, 1),
            # Keep the model's torso axis at the canvas centre.  This is
            # stable for side poses while avoiding an arm-led bounding box.
            "position_x": round(focus_position(body_center_x, fitted_width, target_size[0]), 4),
            "position_y": round(position_y, 4),
            "allow_mirror_extension": False,
            "confidence": round(detection.confidence, 4),
        }
        if target_crop_height_ratio is not None:
            composition["target_crop_height_ratio"] = round(target_crop_height_ratio, 3)
        if product_category in {"shirt", "shoes", "socks"} and protected_region is not None:
            composition["protected_crop_bbox"] = [round(value, 1) for value in protected_region]
        return composition

    @staticmethod
    def _model_view(detection: PersonDetection) -> str:
        keypoints = [point for point in detection.keypoints if point[2] >= BODY_KEYPOINT_CONFIDENCE]
        if len(keypoints) < 4:
            return "detail"
        # A detected hip, knee or ankle is not enough to call a photo a
        # portrait.  Detail shots can contain hands and legs, and must stay
        # on the simple edge-to-edge path.  A model portrait needs a visible
        # facial landmark (nose/eye/ear) according to the product rule.
        has_face = any(detection.keypoints[index][2] >= BODY_KEYPOINT_CONFIDENCE for index in range(0, 5))
        has_feet = len(detection.keypoints) >= 17 and any(
            detection.keypoints[index][2] >= BODY_KEYPOINT_CONFIDENCE for index in (15, 16)
        )
        if has_face and has_feet:
            return "full_body"
        if has_face:
            return "upper_body"
        if has_feet:
            # A shoe/sock source often shows only the lower body. Keep it on
            # the model composition path so feet can be normalized instead
            # of being treated as an arbitrary edge-to-edge detail photo.
            return "lower_body"
        return "detail"

    def compose_for_sizes(
        self,
        source: Path,
        sizes: dict[str, tuple[tuple[int, int], tuple[int, int]]],
        product_category: str | None = None,
    ) -> dict[str, object]:
        with Image.open(source) as opened:
            image_size = opened.size
        detections = self.detect(source)
        visibility = body_visibility(detections, image_size)
        image_width, image_height = image_size
        visibility["feet_clipped"] = any(
            any(
                len(detection.keypoints) > index
                and detection.keypoints[index][2] >= BODY_KEYPOINT_CONFIDENCE
                and (
                    detection.keypoints[index][0] <= image_width * 0.02
                    or detection.keypoints[index][0] >= image_width * 0.98
                    or detection.keypoints[index][1] >= image_height * 0.98
                )
                for index in (15, 16)
            )
            or detection.bbox[3] >= image_height * 0.985
            for detection in detections
        )
        visibility["person_box_height_ratio"] = max(
            ((detection.bbox[3] - detection.bbox[1]) / max(1, image_height))
            for detection in detections
        ) if detections else 0.0
        visibility["person_box_width_ratio"] = max(
            ((detection.bbox[2] - detection.bbox[0]) / max(1, image_width))
            for detection in detections
        ) if detections else 0.0
        if not detections:
            return {
                "model_view": "detail",
                "detector": self.model_path.name,
                "person_count": len(detections),
                "body_visibility": visibility,
            }
        if len(detections) > 1:
            # Multiple people are a model-group candidate, not a detail shot.
            # Use the union box for a conservative, group-safe composition;
            # this keeps the top-most head in frame instead of silently
            # falling back to a detail crop.
            group_bbox = (
                min(item.bbox[0] for item in detections),
                min(item.bbox[1] for item in detections),
                max(item.bbox[2] for item in detections),
                max(item.bbox[3] for item in detections),
            )
            # Preserve the strongest landmark for each keypoint index. The
            # union box alone loses feet/face evidence, preventing
            # category-focused crops (especially shoes) for group photos.
            group_keypoints = []
            max_keypoints = max((len(item.keypoints) for item in detections), default=0)
            for index in range(max_keypoints):
                candidates = [
                    item.keypoints[index]
                    for item in detections
                    if len(item.keypoints) > index
                    and item.keypoints[index][2] >= BODY_KEYPOINT_CONFIDENCE
                ]
                group_keypoints.append(
                    max(candidates, key=lambda point: point[2])
                    if candidates
                    else (0.0, 0.0, 0.0)
                )
            group_detection = PersonDetection(
                group_bbox,
                max(item.confidence for item in detections),
                tuple(group_keypoints),
            )
            shirt_region = None
            if product_category == "shirt":
                regions = [shirt_protected_region(item, image_size) for item in detections]
                shirt_region = (
                    min(region[0] for region in regions),
                    min(region[1] for region in regions),
                    max(region[2] for region in regions),
                    max(region[3] for region in regions),
                )
            shoe_region = None
            if product_category in {"shoes", "socks"}:
                regions = []
                for item in detections:
                    foot_points = [
                        item.keypoints[index]
                        for index in (15, 16)
                        if len(item.keypoints) > index
                        and item.keypoints[index][2] >= BODY_KEYPOINT_CONFIDENCE
                    ]
                    if not foot_points:
                        continue
                    body_height = max(1.0, item.bbox[3] - item.bbox[1])
                    foot_left = min(point[0] for point in foot_points)
                    foot_right = max(point[0] for point in foot_points)
                    foot_width = max(body_height * 0.12, (item.bbox[2] - item.bbox[0]) * 0.16)
                    regions.append((
                        max(0.0, foot_left - foot_width),
                        max(0.0, min(point[1] for point in foot_points) - body_height * 0.16),
                        min(image_width, foot_right + foot_width),
                        min(image_height, item.bbox[3] + body_height * 0.035),
                    ))
                if regions:
                    shoe_region = (
                        min(region[0] for region in regions),
                        min(region[1] for region in regions),
                        max(region[2] for region in regions),
                        max(region[3] for region in regions),
                    )
            by_size = {
                size_name: self._composition(
                    group_detection,
                    image_size,
                    size,
                    "group",
                    product_category,
                    shirt_region or shoe_region,
                )
                for size_name, (size, _background) in sizes.items()
            }
            return {
                "model_view": "group",
                "confidence": round(max(item.confidence for item in detections), 4),
                "detector": self.model_path.name,
                "person_count": len(detections),
                "person_bboxes": [list(item.bbox) for item in detections],
                "body_visibility": visibility,
                "composition_mode": "zoom_focus",
                "by_size": by_size,
            }
        detection = detections[0]
        model_view = self._model_view(detection)
        if model_view == "detail":
            return {
                "model_view": "detail",
                "confidence": round(detection.confidence, 4),
                "detector": self.model_path.name,
                "person_count": 1,
                "person_bboxes": [list(detection.bbox)],
                "body_visibility": visibility,
            }
        by_size = {
            size_name: self._composition(
                detection,
                image_size,
                size,
                model_view,
                product_category,
            )
            for size_name, (size, _background) in sizes.items()
        }
        return {
            "composition_mode": "zoom_focus",
            "model_view": model_view,
            "by_size": by_size,
            "confidence": round(detection.confidence, 4),
            "detector": self.model_path.name,
            "person_count": len(detections),
            "person_bboxes": [list(detection.bbox)],
            "body_visibility": visibility,
        }
