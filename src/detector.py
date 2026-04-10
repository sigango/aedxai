"""Unified object detector wrapper for AED-XAI.

This module provides a working inference wrapper around two detector families:

- YOLOX-S from the official YOLOX repository
- torchvision Faster R-CNN ResNet-50 FPN V2

The wrapper normalizes their outputs into a shared :class:`Detection` dataclass
and exposes helper utilities needed by the rest of the AED-XAI pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence
from urllib.request import urlretrieve

import cv2
import numpy as np
import yaml

from .utils import COCO_CLASSES, bbox_area, get_device, pairwise_iou

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

_MODEL_NAME_ALIASES = {
    "yolox": "yolox-s",
    "yolox-s": "yolox-s",
    "fasterrcnn": "fasterrcnn_resnet50_fpn_v2",
    "fasterrcnn_resnet50_fpn_v2": "fasterrcnn_resnet50_fpn_v2",
}

_COMPLEXITY_LEVELS = ["low", "medium", "high"]

_VISUALIZATION_COLORS: list[tuple[int, int, int]] = [
    (255, 56, 56),
    (56, 117, 255),
    (56, 255, 151),
    (255, 159, 56),
    (170, 56, 255),
    (56, 221, 255),
    (255, 56, 189),
    (255, 219, 56),
]

# torchvision detection outputs use COCO category ids in the 1..90 range, with
# several ids intentionally unused. This mapping projects those ids onto the
# contiguous 0..79 indexing used by the canonical COCO 80-class list.
COCO_91_TO_80: dict[int, int] = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    13: 11,
    14: 12,
    15: 13,
    16: 14,
    17: 15,
    18: 16,
    19: 17,
    20: 18,
    21: 19,
    22: 20,
    23: 21,
    24: 22,
    25: 23,
    27: 24,
    28: 25,
    31: 26,
    32: 27,
    33: 28,
    34: 29,
    35: 30,
    36: 31,
    37: 32,
    38: 33,
    39: 34,
    40: 35,
    41: 36,
    42: 37,
    43: 38,
    44: 39,
    46: 40,
    47: 41,
    48: 42,
    49: 43,
    50: 44,
    51: 45,
    52: 46,
    53: 47,
    54: 48,
    55: 49,
    56: 50,
    57: 51,
    58: 52,
    59: 53,
    60: 54,
    61: 55,
    62: 56,
    63: 57,
    64: 58,
    65: 59,
    67: 60,
    70: 61,
    72: 62,
    73: 63,
    74: 64,
    75: 65,
    76: 66,
    77: 67,
    78: 68,
    79: 69,
    80: 70,
    81: 71,
    82: 72,
    84: 73,
    85: 74,
    86: 75,
    87: 76,
    88: 77,
    89: 78,
    90: 79,
}


@dataclass(slots=True)
class Detection:
    """Normalized object detection result used across the AED-XAI pipeline."""

    bbox: list[int]
    class_id: int
    class_name: str
    confidence: float
    area: int
    relative_size: str
    detection_id: int


@dataclass(slots=True)
class DetectorThresholds:
    """Threshold state used to control detector post-processing behavior."""

    conf_thresh: float
    nms_thresh: float
    max_detections: int


@dataclass(slots=True)
class SceneComplexity:
    """Structured summary of scene complexity statistics."""

    level: str
    detection_count: int
    avg_pairwise_iou: float
    confidence_variance: float


def _require_torch() -> Any:
    """Import torch lazily so the module can still be parsed without it."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for detector inference.") from exc
    return torch


def _require_torchvision_ops() -> tuple[Any, Any]:
    """Import torchvision NMS and box IoU utilities lazily."""
    try:
        from torchvision.ops import batched_nms, box_iou
    except ImportError as exc:
        raise ImportError("torchvision is required for detector inference utilities.") from exc
    return batched_nms, box_iou


def _resolve_path(path: str) -> Path:
    """Resolve a possibly relative path against the project root."""
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path

    candidates = [
        Path.cwd() / raw_path,
        Path(__file__).resolve().parents[1] / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_detector_config(config_path: str) -> dict[str, Any]:
    """Load the detector configuration section from YAML."""
    resolved_path = _resolve_path(config_path)
    with resolved_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    if "detector" not in raw_config:
        raise KeyError(f"Missing 'detector' section in config: {resolved_path}")
    return raw_config["detector"]


def _normalize_model_name(model_name: str) -> str:
    """Normalize detector aliases to canonical model names."""
    normalized = model_name.strip().lower()
    if normalized not in _MODEL_NAME_ALIASES:
        supported = ", ".join(sorted(set(_MODEL_NAME_ALIASES.values())))
        raise ValueError(f"Unsupported model_name '{model_name}'. Supported values: {supported}")
    return _MODEL_NAME_ALIASES[normalized]


def _relative_size_label(area: int, image_area: int) -> str:
    """Map a bounding box area to the project-wide size categories."""
    if image_area <= 0:
        return "small"

    ratio = area / float(image_area)
    if ratio < 0.01:
        return "small"
    if ratio <= 0.10:
        return "medium"
    return "large"


def _clip_bbox(box: Sequence[float], image_height: int, image_width: int) -> list[int]:
    """Clip floating-point XYXY boxes into valid integer image coordinates."""
    x1, y1, x2, y2 = [float(value) for value in box]

    x1_int = int(np.floor(np.clip(x1, 0, max(0, image_width - 1))))
    y1_int = int(np.floor(np.clip(y1, 0, max(0, image_height - 1))))
    x2_int = int(np.ceil(np.clip(x2, x1_int + 1, image_width)))
    y2_int = int(np.ceil(np.clip(y2, y1_int + 1, image_height)))
    return [x1_int, y1_int, x2_int, y2_int]


def _build_detection(
    box: Sequence[float],
    class_id: int,
    confidence: float,
    detection_id: int,
    image_height: int,
    image_width: int,
) -> Detection | None:
    """Construct a validated :class:`Detection` object from raw model output."""
    if class_id < 0 or class_id >= len(COCO_CLASSES):
        return None

    bbox = _clip_bbox(box, image_height=image_height, image_width=image_width)
    area = bbox_area(bbox)
    if area <= 0:
        return None

    return Detection(
        bbox=bbox,
        class_id=int(class_id),
        class_name=COCO_CLASSES[int(class_id)],
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        area=area,
        relative_size=_relative_size_label(area=area, image_area=image_height * image_width),
        detection_id=int(detection_id),
    )


def _finalize_detections(
    candidates: list[tuple[Sequence[float], int, float]],
    image_height: int,
    image_width: int,
    max_detections: int,
) -> list[Detection]:
    """Sort, validate, and re-index raw detections."""
    ordered = sorted(candidates, key=lambda item: item[2], reverse=True)
    finalized: list[Detection] = []

    for box, class_id, confidence in ordered[:max_detections]:
        detection = _build_detection(
            box=box,
            class_id=class_id,
            confidence=confidence,
            detection_id=len(finalized),
            image_height=image_height,
            image_width=image_width,
        )
        if detection is not None:
            finalized.append(detection)

    return finalized


def _letterbox_preprocess(image: np.ndarray, input_size: Sequence[int]) -> tuple[np.ndarray, float]:
    """Resize and pad an RGB image into the YOLOX input canvas.

    YOLOX expects float32 image tensors in the 0..255 range, not normalized
    0..1 inputs, so the fallback path intentionally avoids dividing by 255.
    """
    target_h, target_w = [int(value) for value in input_size]
    image_h, image_w = image.shape[:2]
    scale = min(target_h / float(image_h), target_w / float(image_w))

    resized_w = max(1, int(round(image_w * scale)))
    resized_h = max(1, int(round(image_h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    padded[:resized_h, :resized_w] = resized

    chw = np.transpose(padded.astype(np.float32), (2, 0, 1))
    return chw, scale


def _extract_detector_complexity_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept either the full YAML payload or the detector subsection."""
    if "detector" in config:
        return config["detector"]
    return config


def compute_scene_complexity(detections: list[Detection], config: Mapping[str, Any]) -> str:
    """Compute scene complexity from detection count, overlap, and confidence spread."""
    detector_config = _extract_detector_complexity_config(config)
    complexity_config = detector_config.get("complexity", {})
    count_config = complexity_config.get("detection_count", {})

    low_count = int(count_config.get("low", 3))
    medium_count = int(count_config.get("medium", 8))
    avg_iou_thresh = float(complexity_config.get("avg_iou_thresh", 0.3))
    confidence_var_thresh = float(complexity_config.get("confidence_var_thresh", 0.1))

    detection_count = len(detections)
    if detection_count <= low_count:
        level_index = 0
    elif detection_count <= medium_count:
        level_index = 1
    else:
        level_index = 2

    if detection_count >= 2:
        boxes = np.asarray([detection.bbox for detection in detections], dtype=np.float32)
        try:
            torch = _require_torch()
            _, box_iou = _require_torchvision_ops()
            box_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            iou_matrix = box_iou(box_tensor, box_tensor).detach().cpu().numpy()
        except Exception:
            iou_matrix = pairwise_iou(boxes.astype(int).tolist())

        upper_triangle = np.triu_indices(detection_count, k=1)
        avg_iou = float(iou_matrix[upper_triangle].mean()) if upper_triangle[0].size else 0.0
    else:
        avg_iou = 0.0

    confidences = np.asarray([detection.confidence for detection in detections], dtype=np.float32)
    confidence_variance = float(np.var(confidences)) if confidences.size else 0.0

    if avg_iou > avg_iou_thresh:
        level_index += 1
    if confidence_variance > confidence_var_thresh:
        level_index += 1

    return _COMPLEXITY_LEVELS[min(level_index, len(_COMPLEXITY_LEVELS) - 1)]


class DetectorWrapper:
    """Unified wrapper for YOLOX-S and Faster R-CNN object detectors."""

    def __init__(self, model_name: str = "yolox-s", config_path: str = "config/detector_config.yaml") -> None:
        """Initialize the detector wrapper and resolve runtime settings."""
        self.config_path = str(config_path)
        self.config = _load_detector_config(config_path)
        self.model_name = _normalize_model_name(model_name)

        if self.model_name == self.config["primary"]["name"]:
            self.model_config = dict(self.config["primary"])
        elif self.model_name == self.config["secondary"]["name"]:
            self.model_config = dict(self.config["secondary"])
        else:
            raise ValueError(f"Model '{self.model_name}' not found in detector config.")

        desired_device = str(self.config.get("device", "cuda")).lower()
        torch = _require_torch()
        if desired_device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested in config but unavailable; falling back to CPU.")
            self.device = torch.device("cpu")
        elif desired_device == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = get_device()

        self.default_nms_thresh = float(self.config.get("nms_thresh", 0.45))
        self.default_conf_thresh = float(self.config.get("conf_thresh", 0.25))
        self.max_detections = int(self.config.get("max_detections", 100))

        self.model: torch.nn.Module | None = None
        self._yolox_preproc_fn: Any | None = None
        self._yolox_postprocess_fn: Any | None = None
        self._yolox_exp: Any | None = None

    def load_model(self) -> None:
        """Load the configured pretrained detector on the selected device."""
        if self.model is not None:
            return

        torch = _require_torch()

        if self.model_name == "yolox-s":
            self.model = self._load_yolox_model(torch=torch)
        elif self.model_name == "fasterrcnn_resnet50_fpn_v2":
            self.model = self._load_fasterrcnn_model()
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        self.model.to(self.device)
        self.model.eval()
        logger.info("Loaded detector '%s' on %s", self.model_name, self.device)

    def unload_model(self) -> None:
        """Release model resources and empty the CUDA cache when available."""
        if self.model is None:
            return

        torch = _require_torch()
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def detect(
        self,
        image: np.ndarray,
        nms_thresh: float | None = None,
        conf_thresh: float | None = None,
    ) -> list[Detection]:
        """Run inference on a single RGB image and return normalized detections."""
        self.load_model()
        assert self.model is not None

        image = self._validate_image(image)
        nms_thresh_value = float(self.default_nms_thresh if nms_thresh is None else nms_thresh)
        conf_thresh_value = float(self.default_conf_thresh if conf_thresh is None else conf_thresh)

        if self.model_name == "yolox-s":
            detections = self._detect_yolox(
                image=image,
                nms_thresh=nms_thresh_value,
                conf_thresh=conf_thresh_value,
            )
        else:
            detections = self._detect_fasterrcnn(
                image=image,
                nms_thresh=nms_thresh_value,
                conf_thresh=conf_thresh_value,
            )

        return detections

    def detect_batch(
        self,
        images: list[np.ndarray],
        nms_thresh: float | None = None,
        conf_thresh: float | None = None,
    ) -> list[list[Detection]]:
        """Run inference sequentially over a list of images."""
        return [
            self.detect(image=image, nms_thresh=nms_thresh, conf_thresh=conf_thresh)
            for image in images
        ]

    def get_model(self) -> Any:
        """Return the underlying PyTorch detector model."""
        self.load_model()
        return self.model

    def get_target_layer(self) -> Any:
        """Return the detector backbone layer used by CAM-based XAI methods."""
        self.load_model()
        assert self.model is not None

        if self.model_name == "yolox-s":
            try:
                dark5 = self.model.backbone.backbone.dark5
                children = list(dark5.children())
                return children[-1] if children else dark5
            except AttributeError:
                candidates = [
                    (name, module)
                    for name, module in self.model.named_modules()
                    if "dark5" in name
                ]
                if candidates:
                    return min(candidates, key=lambda item: len(item[0]))[1]
                raise RuntimeError("Unable to locate YOLOX dark5 target layer.")

        try:
            return self.model.backbone.body.layer4[-1]
        except AttributeError as exc:
            raise RuntimeError("Unable to locate Faster R-CNN layer4 target layer.") from exc

    def visualize(self, image: np.ndarray, detections: list[Detection], show_ids: bool = True) -> np.ndarray:
        """Draw detection boxes and labels onto an RGB image."""
        annotated = self._validate_image(image).copy()
        canvas = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        height, width = canvas.shape[:2]

        for index, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.bbox
            color_rgb = _VISUALIZATION_COLORS[index % len(_VISUALIZATION_COLORS)]
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color_bgr, thickness=2)

            if show_ids:
                label = f"#{detection.detection_id} {detection.class_name} {detection.confidence:.2f}"
            else:
                label = f"{detection.class_name} {detection.confidence:.2f}"

            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2,
            )
            bg_left = x1
            bg_top = max(0, y1 - text_height - baseline - 8)
            bg_right = min(width - 1, x1 + text_width + 8)
            bg_bottom = min(height - 1, bg_top + text_height + baseline + 8)

            cv2.rectangle(canvas, (bg_left, bg_top), (bg_right, bg_bottom), color_bgr, thickness=-1)
            cv2.putText(
                canvas,
                label,
                (bg_left + 4, bg_bottom - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )

        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    def forward_with_gradients(self, image: np.ndarray) -> Any:
        """Run a gradient-enabled forward pass for XAI methods such as Grad-CAM."""
        self.load_model()
        assert self.model is not None

        torch = _require_torch()
        image = self._validate_image(image)

        with torch.enable_grad():
            if self.model_name == "yolox-s":
                processed_image, _ = self._preprocess_yolox(image)
                tensor = torch.from_numpy(np.ascontiguousarray(processed_image)).float()
                if tensor.ndim == 3:
                    tensor = tensor.unsqueeze(0)
                tensor = tensor.to(self.device)
                return self.model(tensor)

            tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
            tensor = tensor.to(self.device)
            return self.model([tensor])

    def compute_scene_complexity(self, detections: list[Detection]) -> str:
        """Compute scene complexity using the wrapper's active detector config."""
        return compute_scene_complexity(detections=detections, config=self.config)

    def _load_yolox_model(self, torch: Any) -> Any:
        """Load an official YOLOX-S checkpoint."""
        try:
            from yolox.exp.build import get_exp_by_name
        except ImportError:
            try:
                from yolox.exp import get_exp
            except ImportError as exc:
                raise ImportError(
                    "YOLOX is not installed. Install it with "
                    "'pip install git+https://github.com/Megvii-BaseDetection/YOLOX.git'."
                ) from exc

            try:
                exp = get_exp(None, self.model_name)
            except TypeError:
                exp = get_exp(exp_file=None, exp_name=self.model_name)
        else:
            exp = get_exp_by_name(self.model_name)

        self._yolox_exp = exp

        try:
            from yolox.data.data_augment import preproc as yolox_preproc
        except ImportError:
            yolox_preproc = None

        try:
            from yolox.utils import postprocess as yolox_postprocess
        except ImportError:
            yolox_postprocess = None

        self._yolox_preproc_fn = yolox_preproc
        self._yolox_postprocess_fn = yolox_postprocess

        weights_path = _resolve_path(self.model_config["weights_path"])
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        if not weights_path.exists():
            weights_url = self.model_config.get("weights_url")
            if not weights_url:
                raise FileNotFoundError(f"YOLOX weights missing: {weights_path}")
            logger.info("Downloading YOLOX weights from %s", weights_url)
            urlretrieve(str(weights_url), weights_path)

        model = exp.get_model()
        checkpoint = torch.load(str(weights_path), map_location="cpu")
        state_dict = checkpoint["model"] if isinstance(checkpoint, Mapping) and "model" in checkpoint else checkpoint
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            logger.warning(
                "YOLOX checkpoint loaded with missing keys=%s unexpected keys=%s",
                incompatible.missing_keys,
                incompatible.unexpected_keys,
            )

        if hasattr(model, "head") and hasattr(model.head, "decode_in_inference"):
            model.head.decode_in_inference = True

        return model

    def _load_fasterrcnn_model(self) -> Any:
        """Load torchvision Faster R-CNN ResNet-50 FPN V2."""
        try:
            from torchvision.models.detection import (
                FasterRCNN_ResNet50_FPN_V2_Weights,
                fasterrcnn_resnet50_fpn_v2,
            )
        except ImportError as exc:
            raise ImportError("torchvision detection models are required for Faster R-CNN.") from exc

        weights_name = str(self.model_config.get("weights", "COCO_V1"))
        if hasattr(FasterRCNN_ResNet50_FPN_V2_Weights, weights_name):
            weights = getattr(FasterRCNN_ResNet50_FPN_V2_Weights, weights_name)
        else:
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1

        return fasterrcnn_resnet50_fpn_v2(weights=weights)

    def _detect_fasterrcnn(
        self,
        image: np.ndarray,
        nms_thresh: float,
        conf_thresh: float,
    ) -> list[Detection]:
        """Run torchvision Faster R-CNN inference."""
        torch = _require_torch()
        batched_nms, _ = _require_torchvision_ops()

        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(self.device)

        assert self.model is not None
        with torch.inference_mode():
            raw_outputs = self.model([tensor])
        output = raw_outputs[0]

        boxes = output["boxes"]
        labels = output["labels"]
        scores = output["scores"]

        keep_mask = scores >= conf_thresh
        boxes = boxes[keep_mask]
        labels = labels[keep_mask]
        scores = scores[keep_mask]

        mapped_class_ids: list[int] = []
        kept_indices: list[int] = []
        for index, label in enumerate(labels.detach().cpu().tolist()):
            mapped_class_id = COCO_91_TO_80.get(int(label))
            if mapped_class_id is not None:
                kept_indices.append(index)
                mapped_class_ids.append(mapped_class_id)

        if not kept_indices:
            return []

        index_tensor = torch.as_tensor(kept_indices, dtype=torch.long, device=boxes.device)
        boxes = boxes[index_tensor]
        scores = scores[index_tensor]
        mapped_labels_tensor = torch.as_tensor(mapped_class_ids, dtype=torch.int64, device=boxes.device)

        # Faster R-CNN applies NMS internally, but we intentionally re-run it
        # here so the feedback loop can vary the effective threshold at runtime.
        keep = batched_nms(boxes, scores, mapped_labels_tensor, nms_thresh)[: self.max_detections]
        boxes = boxes[keep].detach().cpu().numpy()
        scores = scores[keep].detach().cpu().numpy()
        class_ids = mapped_labels_tensor[keep].detach().cpu().numpy()

        candidates = [
            (boxes[index].tolist(), int(class_ids[index]), float(scores[index]))
            for index in range(len(scores))
        ]
        return _finalize_detections(
            candidates=candidates,
            image_height=image.shape[0],
            image_width=image.shape[1],
            max_detections=self.max_detections,
        )

    def _detect_yolox(
        self,
        image: np.ndarray,
        nms_thresh: float,
        conf_thresh: float,
    ) -> list[Detection]:
        """Run YOLOX-S inference and normalize outputs."""
        torch = _require_torch()
        batched_nms, _ = _require_torchvision_ops()

        processed_image, scale = self._preprocess_yolox(image)
        tensor = torch.from_numpy(np.ascontiguousarray(processed_image)).float()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)

        assert self.model is not None
        with torch.inference_mode():
            raw_outputs = self.model(tensor)
        predictions = self._postprocess_yolox_outputs(
            raw_outputs=raw_outputs,
            conf_thresh=conf_thresh,
            nms_thresh=nms_thresh,
        )

        if predictions.numel() == 0:
            return []

        # The official YOLOX postprocess path may already filter by confidence
        # and NMS. We intentionally apply a second pass here so AED-XAI can
        # enforce consistent runtime thresholds even if upstream behavior varies.
        scores = predictions[:, 4] * predictions[:, 5]
        class_ids = predictions[:, 6].to(dtype=torch.int64)

        keep = scores >= conf_thresh
        predictions = predictions[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        if predictions.numel() == 0:
            return []

        keep = batched_nms(predictions[:, :4], scores, class_ids, nms_thresh)[: self.max_detections]
        predictions = predictions[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        boxes = (predictions[:, :4] / scale).detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()
        class_ids_np = class_ids.detach().cpu().numpy()

        candidates = [
            (boxes[index].tolist(), int(class_ids_np[index]), float(scores_np[index]))
            for index in range(len(scores_np))
            if 0 <= int(class_ids_np[index]) < len(COCO_CLASSES)
        ]
        return _finalize_detections(
            candidates=candidates,
            image_height=image.shape[0],
            image_width=image.shape[1],
            max_detections=self.max_detections,
        )

    def _preprocess_yolox(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """Prepare an RGB image for YOLOX inference."""
        input_size = self.model_config.get("input_size", [640, 640])
        if self._yolox_preproc_fn is None:
            return _letterbox_preprocess(image=image, input_size=input_size)

        try:
            processed, ratio = self._yolox_preproc_fn(image, input_size)
        except TypeError:
            processed, ratio = self._yolox_preproc_fn(image, input_size, swap=(2, 0, 1))

        processed = np.asarray(processed, dtype=np.float32)
        if processed.ndim == 3 and processed.shape[0] != 3 and processed.shape[-1] == 3:
            processed = np.transpose(processed, (2, 0, 1))
        return processed, float(ratio)

    def _postprocess_yolox_outputs(self, raw_outputs: Any, conf_thresh: float, nms_thresh: float) -> Any:
        """Convert YOLOX raw model outputs into an Nx7 tensor."""
        torch = _require_torch()

        outputs = raw_outputs
        if self._yolox_postprocess_fn is not None:
            try:
                outputs = self._yolox_postprocess_fn(
                    raw_outputs,
                    num_classes=int(self.model_config.get("num_classes", 80)),
                    conf_thre=conf_thresh,
                    nms_thre=nms_thresh,
                )
            except TypeError:
                outputs = self._yolox_postprocess_fn(
                    raw_outputs,
                    int(self.model_config.get("num_classes", 80)),
                    conf_thresh,
                    nms_thresh,
                )

        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]

        if outputs is None:
            return torch.empty((0, 7), dtype=torch.float32, device=self.device)

        if not isinstance(outputs, torch.Tensor):
            outputs = torch.as_tensor(outputs, dtype=torch.float32, device=self.device)

        if outputs.ndim == 3:
            outputs = outputs[0]

        if outputs.numel() == 0:
            return torch.empty((0, 7), dtype=torch.float32, device=self.device)

        if outputs.shape[-1] < 7:
            raise RuntimeError(f"Unexpected YOLOX output shape: {tuple(outputs.shape)}")
        if outputs.shape[-1] > 7:
            raise RuntimeError(
                "YOLOX postprocessing did not produce Nx7 predictions. "
                "Ensure the official YOLOX postprocess utility is available."
            )

        return outputs

    @staticmethod
    def _validate_image(image: np.ndarray) -> np.ndarray:
        """Validate and normalize an image into RGB uint8 HWC format."""
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy array")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must have shape (H, W, 3), got {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image)


ObjectDetector = DetectorWrapper

__all__ = [
    "COCO_91_TO_80",
    "Detection",
    "DetectorThresholds",
    "DetectorWrapper",
    "ObjectDetector",
    "SceneComplexity",
    "compute_scene_complexity",
]
