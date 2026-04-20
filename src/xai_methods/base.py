"""Base classes and shared utilities for AED-XAI explanation methods."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import cv2
import numpy as np

from ..detector import COCO_91_TO_80, Detection

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SaliencyMap:
    """Pixel-level explanation output for a single detection."""

    map: np.ndarray
    method_name: str
    computation_time: float
    detection_id: int

    @property
    def saliency(self) -> np.ndarray:
        """Backward-compatible alias for older code paths."""
        return self.map


class XAIExplainer(ABC):
    """Abstract base class for all XAI explanation methods."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with method-specific config from xai_config.yaml."""
        self.config = dict(config)

    @abstractmethod
    def explain(
        self,
        model: Any,
        image: np.ndarray,
        detection: Detection,
        target_layer: Any | None = None,
    ) -> SaliencyMap:
        """Generate a saliency map for a single detection."""

    def explain_batch(
        self,
        model: Any,
        image: np.ndarray,
        detections: list[Detection],
        target_layer: Any | None = None,
    ) -> list[SaliencyMap]:
        """Default batch implementation that explains detections one by one."""
        return [
            self.explain(model=model, image=image, detection=detection, target_layer=target_layer)
            for detection in detections
        ]

    @staticmethod
    def normalize_saliency(saliency: np.ndarray) -> np.ndarray:
        """Min-max normalize to [0, 1], handling constant maps safely."""
        saliency = np.asarray(saliency, dtype=np.float32)
        if saliency.size == 0:
            return saliency.astype(np.float32)

        smin = float(np.nanmin(saliency))
        smax = float(np.nanmax(saliency))
        if smax - smin < 1e-8:
            return np.zeros_like(saliency, dtype=np.float32)
        return ((saliency - smin) / (smax - smin)).astype(np.float32)


def _require_torch() -> Any:
    """Import torch lazily so the package can still be imported without it."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for XAI explainers.") from exc
    return torch


def _infer_model_family(model: Any) -> str:
    """Infer the detector family from the raw PyTorch module."""
    if hasattr(model, "roi_heads") and hasattr(model, "backbone"):
        return "fasterrcnn"
    if hasattr(model, "head") and hasattr(model, "backbone"):
        return "yolox"
    return "generic"


def _model_device(model: Any) -> Any:
    """Resolve the device used by a PyTorch model."""
    torch = _require_torch()
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _validate_image(image: np.ndarray) -> np.ndarray:
    """Validate and normalize an image to RGB uint8 HWC format."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _letterbox_preprocess(image: np.ndarray, input_size: tuple[int, int] = (640, 640)) -> tuple[np.ndarray, float]:
    """Preprocess an image into YOLOX's expected 0..255 float32 canvas."""
    target_h, target_w = int(input_size[0]), int(input_size[1])
    image_h, image_w = image.shape[:2]
    scale = min(target_h / float(image_h), target_w / float(image_w))

    resized_w = max(1, int(round(image_w * scale)))
    resized_h = max(1, int(round(image_h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    padded[:resized_h, :resized_w] = resized
    chw = np.transpose(padded.astype(np.float32), (2, 0, 1))
    return chw, scale


def _prepare_model_input(model: Any, image: np.ndarray) -> tuple[Any, dict[str, Any]]:
    """Convert an RGB image into the detector-specific input tensor format."""
    torch = _require_torch()
    image = _validate_image(image)
    family = _infer_model_family(model)
    device = _model_device(model)

    if family == "yolox":
        array, scale = _letterbox_preprocess(image)
        tensor = torch.from_numpy(np.ascontiguousarray(array)).float().unsqueeze(0).to(device)
    else:
        tensor = (
            torch.from_numpy(np.ascontiguousarray(image))
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(device)
            / 255.0
        )
        scale = 1.0

    meta = {
        "family": family,
        "scale": float(scale),
        "image_shape": image.shape[:2],
    }
    return tensor, meta


def _forward_detector_from_tensor(model: Any, input_tensor: Any, family: str) -> Any:
    """Run the detector forward pass from a prepared input tensor."""
    if family == "yolox":
        return model(input_tensor)

    try:
        return model([sample for sample in input_tensor])
    except Exception:
        return model(input_tensor)


def _forward_detector(model: Any, image: np.ndarray, require_grad: bool = False) -> tuple[Any, Any, dict[str, Any]]:
    """Preprocess an image and run the detector with or without gradients."""
    torch = _require_torch()
    input_tensor, meta = _prepare_model_input(model, image)
    context = torch.enable_grad() if require_grad else torch.no_grad()
    with context:
        output = _forward_detector_from_tensor(model, input_tensor, meta["family"])
    return output, input_tensor, meta


def _map_label_to_coco(label: int) -> int | None:
    """Map detector label ids into contiguous COCO-80 indices where possible."""
    if label in COCO_91_TO_80:
        return COCO_91_TO_80[label]
    if 0 <= label < 80:
        return label
    if 1 <= label <= 80 and label not in COCO_91_TO_80:
        return label - 1
    return None


def _empty_candidate_tensors(device: Any) -> tuple[Any, Any, Any]:
    """Return empty box, score, and class tensors on the requested device."""
    torch = _require_torch()
    return (
        torch.empty((0, 4), dtype=torch.float32, device=device),
        torch.empty((0,), dtype=torch.float32, device=device),
        torch.empty((0,), dtype=torch.int64, device=device),
    )


def _postprocess_yolox_output(raw_output: Any, device: Any) -> Any:
    """Convert YOLOX raw outputs into an Nx7 tensor in original detector format."""
    torch = _require_torch()
    outputs = raw_output

    if isinstance(outputs, (list, tuple)):
        if not outputs:
            return torch.empty((0, 7), dtype=torch.float32, device=device)
        outputs = outputs[0]

    if outputs is None:
        return torch.empty((0, 7), dtype=torch.float32, device=device)

    if isinstance(outputs, torch.Tensor):
        if outputs.ndim == 3:
            outputs = outputs[0]
        if outputs.ndim == 2 and outputs.shape[-1] == 7:
            return outputs

    try:
        from yolox.utils import postprocess as yolox_postprocess
    except ImportError as exc:
        raise RuntimeError(
            "YOLOX output requires yolox.utils.postprocess to decode detector predictions."
        ) from exc

    outputs = yolox_postprocess(raw_output, num_classes=80, conf_thre=0.0, nms_thre=0.65)
    if isinstance(outputs, (list, tuple)):
        outputs = outputs[0]
    if outputs is None:
        return torch.empty((0, 7), dtype=torch.float32, device=device)
    if outputs.ndim == 3:
        outputs = outputs[0]
    if outputs.shape[-1] != 7:
        raise RuntimeError(f"Unexpected YOLOX output shape after postprocess: {tuple(outputs.shape)}")
    return outputs


def _extract_candidates_from_output(
    output: Any,
    model: Any,
    meta: Mapping[str, Any],
) -> tuple[Any, Any, Any]:
    """Extract detector boxes, scores, and class ids in a common tensor format."""
    torch = _require_torch()
    device = _model_device(model)
    family = str(meta.get("family", _infer_model_family(model)))

    if family == "yolox":
        predictions = _postprocess_yolox_output(output, device=device)
        if predictions.numel() == 0:
            return _empty_candidate_tensors(device)
        boxes = predictions[:, :4] / float(meta.get("scale", 1.0))
        scores = predictions[:, 4] * predictions[:, 5]
        class_ids = predictions[:, 6].to(dtype=torch.int64)
        valid = (class_ids >= 0) & (class_ids < 80)
        if not bool(valid.any().item()):
            return _empty_candidate_tensors(device)
        return boxes[valid], scores[valid], class_ids[valid]

    if isinstance(output, dict):
        output = [output]

    if isinstance(output, (list, tuple)) and output and isinstance(output[0], dict):
        detection_output = output[0]
        boxes = detection_output.get("boxes")
        labels = detection_output.get("labels")
        scores = detection_output.get("scores")
        if boxes is None or labels is None or scores is None:
            return _empty_candidate_tensors(device)

        keep_indices: list[int] = []
        mapped_labels: list[int] = []
        for index in range(int(labels.shape[0])):
            label_value = int(labels[index].detach().item())
            mapped = _map_label_to_coco(label_value)
            if mapped is not None:
                keep_indices.append(index)
                mapped_labels.append(mapped)

        if not keep_indices:
            return _empty_candidate_tensors(boxes.device)

        index_tensor = torch.as_tensor(keep_indices, dtype=torch.long, device=boxes.device)
        return (
            boxes[index_tensor],
            scores[index_tensor],
            torch.as_tensor(mapped_labels, dtype=torch.int64, device=boxes.device),
        )

    return _empty_candidate_tensors(device)


def _iou_to_target(boxes: Any, target_box: Any) -> Any:
    """Compute IoU between N boxes and one target box using differentiable torch ops."""
    torch = _require_torch()
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=boxes.dtype, device=boxes.device)

    x1 = torch.maximum(boxes[:, 0], target_box[0])
    y1 = torch.maximum(boxes[:, 1], target_box[1])
    x2 = torch.minimum(boxes[:, 2], target_box[2])
    y2 = torch.minimum(boxes[:, 3], target_box[3])

    inter_w = torch.clamp(x2 - x1, min=0.0)
    inter_h = torch.clamp(y2 - y1, min=0.0)
    intersection = inter_w * inter_h

    boxes_area = torch.clamp(boxes[:, 2] - boxes[:, 0], min=0.0) * torch.clamp(
        boxes[:, 3] - boxes[:, 1], min=0.0
    )
    target_area = torch.clamp(target_box[2] - target_box[0], min=0.0) * torch.clamp(
        target_box[3] - target_box[1], min=0.0
    )
    union = boxes_area + target_area - intersection
    return torch.where(union > 0, intersection / union, torch.zeros_like(intersection))


def _select_matching_index(
    boxes: Any,
    scores: Any,
    class_ids: Any,
    detection: Detection,
) -> tuple[int | None, float]:
    """Select the model output most closely matching the requested detection."""
    if boxes.numel() == 0 or scores.numel() == 0:
        return None, 0.0

    torch = _require_torch()
    target_box = torch.as_tensor(detection.bbox, dtype=boxes.dtype, device=boxes.device)
    ious = _iou_to_target(boxes, target_box)
    same_class = class_ids == int(detection.class_id)

    if bool(same_class.any().item()) and float(ious[same_class].max().detach().cpu().item()) > 0.0:
        candidate_mask = same_class
    elif float(ious.max().detach().cpu().item()) > 0.0:
        candidate_mask = torch.ones_like(same_class, dtype=torch.bool)
    elif bool(same_class.any().item()):
        candidate_mask = same_class
    else:
        candidate_mask = torch.ones_like(same_class, dtype=torch.bool)

    candidate_indices = torch.arange(boxes.shape[0], device=boxes.device)[candidate_mask]
    ranking = ious[candidate_indices] + (scores[candidate_indices] * 1e-3)
    best_index = candidate_indices[torch.argmax(ranking)]
    return int(best_index.detach().cpu().item()), float(ious[best_index].detach().cpu().item())


def _extract_target_score_from_output(
    output: Any,
    detection: Detection,
    model: Any,
    meta: Mapping[str, Any],
    fallback_tensor: Any,
) -> Any:
    """Extract the scalar confidence score corresponding to a target detection."""
    boxes, scores, class_ids = _extract_candidates_from_output(output, model=model, meta=meta)
    if scores.numel() == 0:
        return fallback_tensor.sum() * 0.0

    best_index, _ = _select_matching_index(boxes, scores, class_ids, detection)
    if best_index is None:
        return fallback_tensor.sum() * 0.0
    return scores[best_index]


def _extract_matching_confidence_from_output(
    output: Any,
    detection: Detection,
    model: Any,
    meta: Mapping[str, Any],
) -> tuple[float, float]:
    """Extract the confidence and IoU of the best-matching output detection."""
    boxes, scores, class_ids = _extract_candidates_from_output(output, model=model, meta=meta)
    if scores.numel() == 0:
        return 0.0, 0.0

    best_index, best_iou = _select_matching_index(boxes, scores, class_ids, detection)
    if best_index is None:
        return 0.0, 0.0
    return float(scores[best_index].detach().cpu().item()), best_iou


def _bbox_prior_map(image_shape: Sequence[int], detection: Detection) -> np.ndarray:
    """Create a normalized Gaussian prior centered on the detection bbox."""
    height, width = int(image_shape[0]), int(image_shape[1])
    x1, y1, x2, y2 = [int(value) for value in detection.bbox]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    sigma_x = max(1.0, (x2 - x1) / 2.0)
    sigma_y = max(1.0, (y2 - y1) / 2.0)

    grid_y, grid_x = np.mgrid[0:height, 0:width]
    prior = np.exp(
        -(
            ((grid_x - cx) ** 2) / (2.0 * sigma_x**2)
            + ((grid_y - cy) ** 2) / (2.0 * sigma_y**2)
        )
    ).astype(np.float32)
    return XAIExplainer.normalize_saliency(prior)


def _finalize_saliency_map(saliency: np.ndarray, image_shape: Sequence[int], detection: Detection) -> np.ndarray:
    """Resize, sanitize, and normalize a saliency map with a bbox-centered fallback."""
    saliency = np.asarray(saliency, dtype=np.float32)
    saliency = np.nan_to_num(saliency, nan=0.0, posinf=0.0, neginf=0.0)

    target_height, target_width = int(image_shape[0]), int(image_shape[1])
    if saliency.shape != (target_height, target_width):
        saliency = cv2.resize(saliency, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    saliency = XAIExplainer.normalize_saliency(saliency)
    if float(saliency.max()) <= 0.0:
        saliency = _bbox_prior_map((target_height, target_width), detection)
    return saliency.astype(np.float32)


def _renormalize_in_bbox(
    saliency: np.ndarray,
    bbox: Sequence[int],
    zero_outside: bool = False,
) -> np.ndarray:
    """Rescale saliency so values inside the bbox span [0, 1].

    Adapted from Jacob Gilpin's pytorch-gradcam-book "Renormalizing the CAMs
    inside every bounding box" trick. In soft mode, only the in-bbox region is
    rescaled; outside values remain on their incoming (already-normalized [0, 1])
    scale. In strict mode, outside is zeroed (PG/EBPG will become trivially 1.0).
    """
    saliency = np.asarray(saliency, dtype=np.float32)
    height, width = saliency.shape[:2]

    x1, y1, x2, y2 = [int(value) for value in bbox]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return saliency

    out = saliency.copy()
    inside = out[y1:y2, x1:x2]
    imin = float(inside.min())
    imax = float(inside.max())
    span = imax - imin
    if span < 1e-8:
        out[y1:y2, x1:x2] = 0.0
    else:
        out[y1:y2, x1:x2] = (inside - imin) / span

    if zero_outside:
        mask = np.zeros_like(out)
        mask[y1:y2, x1:x2] = 1.0
        out = out * mask

    return np.clip(out, 0.0, 1.0).astype(np.float32)


__all__ = ["SaliencyMap", "XAIExplainer"]
