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


def find_onnx_model(model_dir: str | Path) -> Path | None:
    path = Path(model_dir).expanduser()
    if not path.is_dir():
        return None
    models = sorted(path.glob("*.onnx"))
    return models[0] if models else None


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
    def inspect(model_dir: str | Path) -> tuple[bool, str]:
        model_path = find_onnx_model(model_dir)
        if model_path is None:
            return False, "未检测到 ONNX 模型文件"
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
    ) -> dict[str, object]:
        image_width, image_height = image_size
        x1, y1, x2, y2 = detection.bbox
        keypoints = [point for point in detection.keypoints if point[2] >= 0.28]
        if keypoints:
            x1 = min(x1, min(point[0] for point in keypoints))
            y1 = min(y1, min(point[1] for point in keypoints))
            x2 = max(x2, max(point[0] for point in keypoints))
            y2 = max(y2, max(point[1] for point in keypoints))
        has_feet = len(detection.keypoints) >= 17 and any(
            detection.keypoints[index][2] >= 0.28 for index in (15, 16)
        )
        has_head_or_upper_body = len(detection.keypoints) >= 11 and any(
            detection.keypoints[index][2] >= 0.28 for index in range(0, 7)
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
        body_center_x = ((x1 + x2) / 2) / image_width
        # Use the torso axis when pose landmarks are available.  The full
        # person box includes extended arms, so its midpoint can pull an
        # otherwise centred model too far to one side.
        torso_centres = []
        for left_index, right_index in ((5, 6), (11, 12)):
            if (
                len(detection.keypoints) > right_index
                and detection.keypoints[left_index][2] >= 0.28
                and detection.keypoints[right_index][2] >= 0.28
            ):
                torso_centres.append(
                    (detection.keypoints[left_index][0] + detection.keypoints[right_index][0]) / 2
                )
        if torso_centres:
            body_center_x = (sum(torso_centres) / len(torso_centres)) / image_width
        body_center_y = ((y1 + y2) / 2) / image_height
        if partial_upper_body:
            # Square outputs should feature the jacket and face; portrait
            # outputs retain a little more of the lower garment.
            target_ratio = target_size[0] / max(1, target_size[1])
            body_center_y = 0.40 if target_ratio >= 0.9 else 0.46
        elif partial_lower_body:
            body_center_y = 0.56
        base_scale = max(target_size[0] / image_width, target_size[1] / image_height)
        if model_view == "full_body" and product_category in {
            "shirt", "pants", "long_pants", "shorts",
        }:
            body_height = max(1.0, y2 - y1)
            if product_category == "shirt":
                # Bias slightly above the chest so the head is not clipped
                # when a full-body source becomes an upper-body main image.
                body_center_y = (y1 + body_height * 0.28) / image_height
            else:
                body_center_y = (y1 + body_height * 0.70) / image_height
            # Keep the selected garment as the dominant subject.  This crops
            # a full-body source to roughly head-to-waist (or hip-to-shoe)
            # instead of leaving a small person inside a tall canvas.
            # For full-body images the apparel needs to read as the subject,
            # not as a small person in a tall source.  48% retains enough
            # shoulder/hip context while making the chosen half fill a
            # portrait main-image canvas.
            desired_crop_height = image_height * 0.48
            desired_scale = target_size[1] / desired_crop_height
            scale_adjustment = max(0.0, desired_scale / base_scale - 1.0) * 100
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
            if point[2] >= 0.28
        ]
        head_top = min([y1, *(point[1] for point in face_points)])
        safe_crop_top = max(0.0, head_top - image_height * 0.02) * fitted_scale
        vertical_excess = max(0.0, fitted_height - target_size[1])
        max_position_y = (
            self._clamp(safe_crop_top / vertical_excess)
            if vertical_excess > 0
            else 0.5
        )
        position_y = min(
            focus_position(body_center_y, fitted_height, target_size[1]),
            max_position_y,
        )

        return {
            "composition_mode": "zoom_focus",
            "scale_adjustment_percent": round(scale_adjustment, 1),
            # Keep the model's torso axis at the canvas centre.  This is
            # stable for side poses while avoiding an arm-led bounding box.
            "position_x": round(focus_position(body_center_x, fitted_width, target_size[0]), 4),
            "position_y": round(position_y, 4),
            "allow_mirror_extension": False,
            "confidence": round(detection.confidence, 4),
        }

    @staticmethod
    def _model_view(detection: PersonDetection) -> str:
        keypoints = [point for point in detection.keypoints if point[2] >= 0.28]
        if len(keypoints) < 4:
            return "detail"
        # A detected hip, knee or ankle is not enough to call a photo a
        # portrait.  Detail shots can contain hands and legs, and must stay
        # on the simple edge-to-edge path.  A model portrait needs a visible
        # facial landmark (nose/eye/ear) according to the product rule.
        has_face = any(detection.keypoints[index][2] >= 0.28 for index in range(0, 5))
        has_feet = len(detection.keypoints) >= 17 and any(
            detection.keypoints[index][2] >= 0.28 for index in (15, 16)
        )
        if has_face and has_feet:
            return "full_body"
        if has_face:
            return "upper_body"
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
        if not detections or len(detections) > 1:
            return {
                "model_view": "detail",
                "detector": self.model_path.name,
                "person_count": len(detections),
            }
        detection = detections[0]
        model_view = self._model_view(detection)
        if model_view == "detail":
            return {
                "model_view": "detail",
                "confidence": round(detection.confidence, 4),
                "detector": self.model_path.name,
                "person_count": 1,
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
        }
