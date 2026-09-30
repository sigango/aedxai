"""Per-detection evidence features for the explanation-as-evidence experiments.

Research question: do explanations carry information about whether a detection
is correct that output-based and gradient-based uncertainty do not? Each feature
family below maps to one baseline or treatment in ``scripts/analyze_evidence.py``:

- ``output_features``      MetaDetect-style box/score statistics (baseline).
- ``gradient_features``    gradient-uncertainty statistics in the spirit of
                           Riedlinger et al. (WACV 2023) (strong baseline).
- ``map_features``         structure of one saliency map relative to the box.
- ``curve_features``       shape of the insertion/deletion confidence curves.
- ``agreement_features``   agreement between two explainers' maps.

All functions are pure NumPy except ``gradient_features``, which needs torch.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any, Sequence

import cv2
import numpy as np

if TYPE_CHECKING:
    from .detector import Detection


def _box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def _clip_box(bbox: Sequence[float], shape: Sequence[int]) -> tuple[int, int, int, int]:
    height, width = int(shape[0]), int(shape[1])
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def _scaled_box(bbox: Sequence[float], scale: float, shape: Sequence[int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w, half_h = (x2 - x1) * scale / 2.0, (y2 - y1) * scale / 2.0
    return _clip_box([cx - half_w, cy - half_h, cx + half_w, cy + half_h], shape)


def output_features(
    detection: "Detection",
    all_detections: Sequence["Detection"],
    image: np.ndarray,
) -> dict[str, float]:
    """Box/score statistics available before any explanation (MetaDetect-lite)."""
    from .xai_selector import compute_image_entropy

    height, width = image.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in detection.bbox]
    box_w, box_h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    others = [other for other in all_detections if other.detection_id != detection.detection_id]
    ious_any = [_box_iou(detection.bbox, other.bbox) for other in others]
    ious_same = [
        _box_iou(detection.bbox, other.bbox) for other in others if other.class_id == detection.class_id
    ]
    higher = sum(1 for other in others if other.confidence > detection.confidence)
    margin = 2
    return {
        "confidence": float(detection.confidence),
        "log_area": float(math.log(box_w * box_h)),
        "rel_area": float(box_w * box_h / max(1.0, float(height * width))),
        "aspect_ratio": float(box_w / box_h),
        "num_detections": float(len(all_detections)),
        "max_iou_any": float(max(ious_any, default=0.0)),
        "max_iou_same_class": float(max(ious_same, default=0.0)),
        "num_overlapping": float(sum(1 for value in ious_any if value > 0.3)),
        "confidence_rank": float(higher / max(1, len(all_detections) - 1)) if len(all_detections) > 1 else 0.0,
        "touches_border": float(x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin),
        "box_entropy": float(compute_image_entropy(image, detection.bbox)),
    }


def gradient_features(
    model: Any,
    image: np.ndarray,
    detection: "Detection",
    target_layer: Any,
) -> dict[str, float]:
    """Gradient statistics of the detection's score w.r.t. a backbone layer.

    One forward+backward pass. Records norms of the activation gradient, of
    activation*gradient, and of the gradient w.r.t. the layer's own parameters.
    """
    from .xai_methods.base import (
        _extract_target_score_from_output,
        _forward_detector_from_tensor,
        _prepare_model_input,
        _require_torch,
    )

    torch = _require_torch()
    t_start = time.time()
    input_tensor, meta = _prepare_model_input(model, image)
    captured: dict[str, Any] = {}

    def _forward_hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured["act"] = output

    def _backward_hook(_module: Any, _grad_input: Any, grad_output: Any) -> None:
        if grad_output:
            captured["grad"] = grad_output[0]

    handle_fwd = target_layer.register_forward_hook(_forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(_backward_hook)
    try:
        if hasattr(model, "zero_grad"):
            model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            output = _forward_detector_from_tensor(model, input_tensor, str(meta.get("family", "generic")))
            score = _extract_target_score_from_output(
                output=output,
                detection=detection,
                model=model,
                meta=meta,
                fallback_tensor=input_tensor,
            )
            if not score.requires_grad:
                return _empty_gradient_features(float(score.detach().cpu().item()), time.time() - t_start)
            score.backward()

        grad = captured.get("grad")
        act = captured.get("act")
        if grad is None or act is None:
            return _empty_gradient_features(float(score.detach().cpu().item()), time.time() - t_start)
        grad = grad.detach().float()
        act = act.detach().float()
        param_sq = 0.0
        for parameter in target_layer.parameters():
            if parameter.grad is not None:
                param_sq += float(parameter.grad.detach().float().pow(2).sum().cpu().item())
        return {
            "grad_score": float(score.detach().cpu().item()),
            "grad_l2": float(grad.norm(p=2).cpu().item()),
            "grad_abs_mean": float(grad.abs().mean().cpu().item()),
            "grad_max": float(grad.abs().max().cpu().item()),
            "grad_act_l2": float((grad * act).norm(p=2).cpu().item()),
            "grad_param_l2": float(math.sqrt(param_sq)),
            "grad_time": float(time.time() - t_start),
        }
    finally:
        handle_fwd.remove()
        handle_bwd.remove()
        if hasattr(model, "zero_grad"):
            model.zero_grad(set_to_none=True)


def _empty_gradient_features(score: float, elapsed: float) -> dict[str, float]:
    return {
        "grad_score": score,
        "grad_l2": 0.0,
        "grad_abs_mean": 0.0,
        "grad_max": 0.0,
        "grad_act_l2": 0.0,
        "grad_param_l2": 0.0,
        "grad_time": float(elapsed),
    }


def map_features(saliency: np.ndarray, bbox: Sequence[float], context_scale: float = 2.0) -> dict[str, float]:
    """Where a saliency map puts its mass relative to the detection box."""
    array = np.clip(np.asarray(saliency, dtype=np.float64), 0.0, None)
    total = float(array.sum())
    height, width = array.shape
    bx1, by1, bx2, by2 = _clip_box(bbox, array.shape)
    cx1, cy1, cx2, cy2 = _scaled_box(bbox, context_scale, array.shape)
    box_area = max(1, (bx2 - bx1) * (by2 - by1))
    if total <= 1e-12:
        return {
            "mass_in_box": 0.0,
            "mass_in_context_ring": 0.0,
            "mass_outside_context": 0.0,
            "peak_in_box": 0.0,
            "com_offset": 1.0,
            "map_entropy": 0.0,
            "topk_in_box": 0.0,
        }

    in_box = float(array[by1:by2, bx1:bx2].sum())
    in_context = float(array[cy1:cy2, cx1:cx2].sum())
    peak_y, peak_x = np.unravel_index(int(np.argmax(array)), array.shape)
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    com_x = float((array * grid_x).sum() / total)
    com_y = float((array * grid_y).sum() / total)
    box_cx, box_cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    diag = max(1.0, math.hypot(bx2 - bx1, by2 - by1))
    probabilities = array.reshape(-1) / total
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum() / math.log(max(2, probabilities.size)))

    k = min(array.size, box_area)
    top_indices = np.argpartition(array.reshape(-1), -k)[-k:]
    top_y, top_x = np.unravel_index(top_indices, array.shape)
    topk_in_box = float(np.mean((top_x >= bx1) & (top_x < bx2) & (top_y >= by1) & (top_y < by2)))

    return {
        "mass_in_box": in_box / total,
        "mass_in_context_ring": max(0.0, in_context - in_box) / total,
        "mass_outside_context": max(0.0, total - in_context) / total,
        "peak_in_box": float(bx1 <= peak_x < bx2 and by1 <= peak_y < by2),
        "com_offset": float(math.hypot(com_x - box_cx, com_y - box_cy) / diag),
        "map_entropy": entropy,
        "topk_in_box": topk_in_box,
    }


def curve_features(
    insertion_curve: Sequence[float],
    deletion_curve: Sequence[float],
    prefix: str = "",
) -> dict[str, float]:
    """Shape statistics of insertion/deletion curves, normalised by the unperturbed score."""
    insertion = np.asarray(insertion_curve, dtype=np.float64)
    deletion = np.asarray(deletion_curve, dtype=np.float64)
    if insertion.size < 2 or deletion.size < 2:
        return {}
    reference = max(float(deletion[0]), 1e-6)  # score of the unperturbed image
    steps = deletion.size - 1
    below_half = np.nonzero(deletion <= 0.5 * reference)[0]
    above_half = np.nonzero(insertion >= 0.5 * reference)[0]
    return {
        f"{prefix}ref_score": float(deletion[0]),
        f"{prefix}del_first_drop": float((deletion[0] - deletion[1]) / reference),
        f"{prefix}ins_first_gain": float(insertion[1] / reference),
        f"{prefix}del_half_step": float(below_half[0] / steps) if below_half.size else 1.0,
        f"{prefix}ins_half_step": float(above_half[0] / steps) if above_half.size else 1.0,
        f"{prefix}del_auc_norm": float(np.mean((deletion[:-1] + deletion[1:]) / 2.0) / reference),
        f"{prefix}ins_auc_norm": float(np.mean((insertion[:-1] + insertion[1:]) / 2.0) / reference),
        f"{prefix}del_monotone": float(np.mean(np.diff(deletion) <= 1e-6)),
    }


def agreement_features(
    map_a: np.ndarray,
    map_b: np.ndarray,
    bbox: Sequence[float],
    resolution: int = 64,
    top_fraction: float = 0.1,
) -> dict[str, float]:
    """Agreement between two saliency maps: rank correlation and top-k overlap."""
    a = cv2.resize(np.asarray(map_a, dtype=np.float32), (resolution, resolution), interpolation=cv2.INTER_AREA)
    b = cv2.resize(np.asarray(map_b, dtype=np.float32), (resolution, resolution), interpolation=cv2.INTER_AREA)
    rank_corr = _spearman(a.reshape(-1), b.reshape(-1))

    k = max(1, int(round(top_fraction * a.size)))
    top_a = set(np.argpartition(a.reshape(-1), -k)[-k:].tolist())
    top_b = set(np.argpartition(b.reshape(-1), -k)[-k:].tolist())
    top_iou = len(top_a & top_b) / max(1, len(top_a | top_b))

    height, width = np.asarray(map_a).shape[:2]
    bx1, by1, bx2, by2 = _clip_box(bbox, (height, width))
    sx, sy = resolution / float(width), resolution / float(height)
    rx1, ry1 = int(bx1 * sx), int(by1 * sy)
    rx2, ry2 = max(rx1 + 1, int(math.ceil(bx2 * sx))), max(ry1 + 1, int(math.ceil(by2 * sy)))
    in_box_corr = _spearman(a[ry1:ry2, rx1:rx2].reshape(-1), b[ry1:ry2, rx1:rx2].reshape(-1))
    return {"rank_corr": rank_corr, "top10_iou": float(top_iou), "in_box_rank_corr": in_box_corr}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation with average ranks for ties; 0 for constant inputs."""
    if x.size < 3:
        return 0.0
    from scipy.stats import rankdata

    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


__all__ = [
    "agreement_features",
    "curve_features",
    "gradient_features",
    "map_features",
    "output_features",
]
