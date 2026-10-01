"""Per-detection evidence features for the explanation-as-evidence experiments.

Research question: do explanations carry information about whether a detection
is correct that output-based and gradient-based uncertainty do not? Each feature
family below maps to one baseline or treatment in ``scripts/analyze_evidence.py``:

- ``output_features``      post-NMS box/score statistics (MetaDetect-lite baseline).
- ``prenms_features``      statistics of the pre-NMS candidates around the box
                           (MetaDetect, Schubert et al. 2021) (strong baseline).
- ``gradient_features``    score-gradient norms at the CAM layer, plus norms of the
                           self-learning loss gradient at the detection head in the
                           spirit of Riedlinger et al. (WACV 2023) (strong baseline).
- ``tta_features``         stability of the detection under test-time augmentation
                           (strong, compute-matched baseline).
- ``map_features``         structure of one saliency map relative to the box.
- ``curve_features``       shape of the insertion/deletion confidence curves.
- ``agreement_features``   agreement between two explainers' maps.

All functions are pure NumPy except ``gradient_features`` and ``run_tta``, which
need torch and a detector.
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


def _boxes_iou(boxes: np.ndarray, box: Sequence[float]) -> np.ndarray:
    """IoU of N boxes (N x 4, xyxy) with one box."""
    if boxes.size == 0:
        return np.zeros(0, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    x1, y1, x2, y2 = [float(value) for value in box]
    ix1, iy1 = np.maximum(boxes[:, 0], x1), np.maximum(boxes[:, 1], y1)
    ix2, iy2 = np.minimum(boxes[:, 2], x2), np.minimum(boxes[:, 3], y2)
    intersection = np.clip(ix2 - ix1, 0.0, None) * np.clip(iy2 - iy1, 0.0, None)
    areas = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    union = areas + max(0.0, x2 - x1) * max(0.0, y2 - y1) - intersection
    return np.where(union > 0, intersection / np.maximum(union, 1e-12), 0.0)


def candidate_arrays(
    candidates: Sequence[tuple[Sequence[float], int, float]] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(boxes N x 4, class ids N, scores N) from ``DetectorWrapper.detect_candidates`` output."""
    if not candidates:
        return np.zeros((0, 4)), np.zeros(0, dtype=np.int64), np.zeros(0)
    boxes = np.asarray([list(box) for box, _, _ in candidates], dtype=np.float64).reshape(-1, 4)
    classes = np.asarray([int(class_id) for _, class_id, _ in candidates], dtype=np.int64)
    scores = np.asarray([float(score) for _, _, score in candidates], dtype=np.float64)
    return boxes, classes, scores


def prenms_features(
    detection: "Detection",
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """MetaDetect-style statistics of the pre-NMS candidates that overlap the detection.

    ``boxes``/``classes``/``scores`` are every candidate the detector produced
    before non-maximum suppression (``DetectorWrapper.detect_candidates``).
    The candidate set includes the detection's own pre-NMS box.
    """
    x1, y1, x2, y2 = [float(value) for value in detection.bbox]
    box_w, box_h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    ious = _boxes_iou(boxes, detection.bbox)
    same = classes == int(detection.class_id)
    near = same & (ious >= iou_threshold)
    other = (~same) & (ious >= iou_threshold)
    count = int(near.sum())
    near_scores, near_ious, near_boxes = scores[near], ious[near], boxes[near]
    if count >= 2:
        scale = np.array([box_w, box_h, box_w, box_h])
        coord_std = float(np.mean(np.std(near_boxes, axis=0) / scale))
        areas = np.clip(near_boxes[:, 2] - near_boxes[:, 0], 1.0, None) * np.clip(near_boxes[:, 3] - near_boxes[:, 1], 1.0, None)
        logarea_std = float(np.std(np.log(areas)))
    else:
        coord_std, logarea_std = 0.0, 0.0
    other_max = float(scores[other].max()) if other.any() else 0.0
    return {
        "count": float(count),
        "count_iou03": float((same & (ious >= 0.3)).sum()),
        "score_sum": float(near_scores.sum()),
        "score_mean": float(near_scores.mean()) if count else 0.0,
        "score_std": float(near_scores.std()) if count else 0.0,
        "score_max": float(near_scores.max()) if count else 0.0,
        "score_min": float(near_scores.min()) if count else 0.0,
        "iou_mean": float(near_ious.mean()) if count else 0.0,
        "iou_std": float(near_ious.std()) if count else 0.0,
        "iou_min": float(near_ious.min()) if count else 0.0,
        "score_weighted_iou": float((near_scores * near_ious).sum() / max(near_scores.sum(), 1e-12)) if count else 0.0,
        "coord_std": coord_std,
        "logarea_std": logarea_std,
        "other_class_count": float(other.sum()),
        "other_class_max_score": other_max,
        "other_class_score_ratio": float(other_max / max(float(detection.confidence), 1e-6)),
    }


# ----------------------------------------------------------------------------- test-time augmentation

TTA_VIEWS = ("hflip", "shrink", "shift", "dark")
TTA_FILL = 114
TTA_SHRINK = 0.75
TTA_SHIFT = 0.06
TTA_DARK = 0.6


def tta_view(image: np.ndarray, name: str) -> tuple[np.ndarray, dict[str, float]]:
    """Augmented copy of ``image`` and the parameters needed to map boxes back.

    The views change what the detector sees even though every detector resizes
    its input: ``shrink`` pastes a 0.75x copy onto a neutral canvas of the
    original size (objects get smaller), ``shift`` translates by 6% of each side,
    ``hflip`` mirrors, and ``dark`` scales intensities by 0.6.
    """
    height, width = image.shape[:2]
    if name == "hflip":
        return np.ascontiguousarray(image[:, ::-1]), {"width": float(width)}
    if name == "shrink":
        new_w, new_h = max(1, int(round(width * TTA_SHRINK))), max(1, int(round(height * TTA_SHRINK)))
        canvas = np.full_like(image, TTA_FILL)
        ox, oy = (width - new_w) // 2, (height - new_h) // 2
        canvas[oy : oy + new_h, ox : ox + new_w] = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return canvas, {"sx": new_w / float(width), "sy": new_h / float(height), "ox": float(ox), "oy": float(oy)}
    if name == "shift":
        dx, dy = int(round(width * TTA_SHIFT)), int(round(height * TTA_SHIFT))
        canvas = np.full_like(image, TTA_FILL)
        canvas[dy:, dx:] = image[: height - dy, : width - dx]
        return canvas, {"dx": float(dx), "dy": float(dy)}
    if name == "dark":
        return np.clip(image.astype(np.float32) * TTA_DARK, 0, 255).astype(np.uint8), {}
    raise ValueError(f"Unknown TTA view '{name}'. Available: {TTA_VIEWS}")


def tta_map_back(boxes: np.ndarray, name: str, params: dict[str, float]) -> np.ndarray:
    """Map boxes detected on a TTA view back into original-image coordinates."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    if boxes.size == 0:
        return boxes
    if name == "hflip":
        width = params["width"]
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
    elif name == "shrink":
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - params["ox"]) / params["sx"]
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - params["oy"]) / params["sy"]
    elif name == "shift":
        boxes[:, [0, 2]] -= params["dx"]
        boxes[:, [1, 3]] -= params["dy"]
    return boxes


def run_tta(
    detector: Any,
    image: np.ndarray,
    views: Sequence[str] = TTA_VIEWS,
    conf_thresh: float = 0.05,
) -> list[dict[str, Any]]:
    """Detect on every TTA view (one forward each); boxes mapped back to the original image."""
    outputs = []
    for name in views:
        view, params = tta_view(image, name)
        detections = detector.detect(view, conf_thresh=conf_thresh)
        boxes = np.asarray([det.bbox for det in detections], dtype=np.float64).reshape(-1, 4)
        outputs.append(
            {
                "name": name,
                "boxes": tta_map_back(boxes, name, params),
                "classes": np.asarray([det.class_id for det in detections], dtype=np.int64),
                "scores": np.asarray([det.confidence for det in detections], dtype=np.float64),
            }
        )
    return outputs


def tta_features(
    detection: "Detection",
    tta_outputs: Sequence[dict[str, Any]],
    match_iou: float = 0.3,
) -> dict[str, float]:
    """How the detection's score, box and class hold up across TTA views.

    Per view: the same-class detection with the highest IoU is the match; its
    score counts only if IoU >= ``match_iou``. A class flip is a different-class
    detection with IoU >= 0.5 that outscores the same-class match.
    """
    confidence = max(float(detection.confidence), 1e-6)
    scores, ious, flips = [], [], []
    features: dict[str, float] = {}
    for output in tta_outputs:
        boxes, classes, view_scores = output["boxes"], output["classes"], output["scores"]
        all_ious = _boxes_iou(boxes, detection.bbox)
        same = classes == int(detection.class_id)
        if same.any():
            index = int(np.argmax(np.where(same, all_ious, -1.0)))
            iou = float(all_ious[index])
            score = float(view_scores[index]) if iou >= match_iou else 0.0
        else:
            iou, score = 0.0, 0.0
        rivals = (~same) & (all_ious >= 0.5)
        flips.append(float(rivals.any() and float(view_scores[rivals].max()) > score))
        scores.append(score)
        ious.append(iou)
        features[f"score_{output['name']}"] = score
        features[f"iou_{output['name']}"] = iou
    if not scores:
        return {}
    score_array, iou_array = np.asarray(scores), np.asarray(ious)
    features.update(
        {
            "score_mean": float(score_array.mean()),
            "score_min": float(score_array.min()),
            "score_std": float(score_array.std()),
            "score_ratio": float(score_array.mean() / confidence),
            "iou_mean": float(iou_array.mean()),
            "iou_min": float(iou_array.min()),
            "match_rate": float(np.mean(iou_array >= 0.5)),
            "class_flip_rate": float(np.mean(flips)),
        }
    )
    return features


# ----------------------------------------------------------------------------- gradients

# Parameter groups for the Riedlinger-style gradient features, per detector family.
# Paths that do not exist in a given model are skipped.
GRADIENT_GROUPS: dict[str, dict[str, tuple[str, ...]]] = {
    "yolox": {
        "cls_pred": ("head.cls_preds",),
        "obj_pred": ("head.obj_preds",),
        "cls_convs": ("head.cls_convs",),
        "reg_convs": ("head.reg_convs",),
        "stems": ("head.stems",),
        "neck": ("backbone.C3_p3", "backbone.C3_n3", "backbone.C3_n4"),
    },
    "fasterrcnn": {
        "cls_score": ("roi_heads.box_predictor.cls_score",),
        "box_head": ("roi_heads.box_head",),
        "fpn": ("backbone.fpn",),
        "layer4": ("backbone.body.layer4",),
    },
}


def _group_gradient_stats(model: Any, groups: dict[str, Sequence[str]], score: float) -> dict[str, float]:
    """Norms of the self-learning loss gradient, -log(score), per parameter group.

    d(-log s)/dθ = -(1/s) ds/dθ, so the score gradient from the single backward
    pass is rescaled by 1/s. The box-regression term of the loss is omitted: with
    the detection as its own pseudo-label it is zero for the post-NMS box.
    """
    from .xai_methods.base import _require_torch

    torch = _require_torch()
    scale = 1.0 / max(float(score), 1e-6)
    features: dict[str, float] = {}
    for name, paths in groups.items():
        grads = []
        for path in paths:
            try:
                module = model.get_submodule(path)
            except AttributeError:
                continue
            grads += [p.grad.detach().float().reshape(-1) for p in module.parameters() if p.grad is not None]
        if not grads:
            continue
        flat = torch.cat(grads)
        features[f"rg_{name}_l2"] = float(flat.norm(p=2).cpu().item() * scale)
        features[f"rg_{name}_l1"] = float(flat.abs().mean().cpu().item() * scale)
        features[f"rg_{name}_max"] = float(flat.abs().max().cpu().item() * scale)
        features[f"rg_{name}_std"] = float(flat.std().cpu().item() * scale) if flat.numel() > 1 else 0.0
    return features


def gradient_features(
    model: Any,
    image: np.ndarray,
    detection: "Detection",
    target_layer: Any,
    param_groups: dict[str, Sequence[str]] | None = None,
) -> dict[str, float]:
    """Gradient statistics of the detection's score (one forward + backward pass).

    ``grad_*``: norms of the activation gradient, of activation*gradient and of
    the gradient w.r.t. the CAM target layer's own parameters (lite baseline).
    ``rg_<group>_*``: Riedlinger-style norms of the self-learning loss gradient
    w.r.t. the detection-head parameters (``GRADIENT_GROUPS`` by detector
    family unless ``param_groups`` is given).
    """
    from .xai_methods.base import (
        _extract_target_score_from_output,
        _forward_detector_from_tensor,
        _infer_model_family,
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
        score_value = float(score.detach().cpu().item())
        groups = param_groups if param_groups is not None else GRADIENT_GROUPS.get(_infer_model_family(model), {})
        features = {
            "grad_score": score_value,
            "grad_l2": float(grad.norm(p=2).cpu().item()),
            "grad_abs_mean": float(grad.abs().mean().cpu().item()),
            "grad_max": float(grad.abs().max().cpu().item()),
            "grad_act_l2": float((grad * act).norm(p=2).cpu().item()),
            "grad_param_l2": float(math.sqrt(param_sq)),
        }
        features.update(_group_gradient_stats(model, dict(groups), score_value))
        features["grad_time"] = float(time.time() - t_start)
        return features
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
    "GRADIENT_GROUPS",
    "TTA_VIEWS",
    "agreement_features",
    "candidate_arrays",
    "curve_features",
    "gradient_features",
    "map_features",
    "output_features",
    "prenms_features",
    "run_tta",
    "tta_features",
    "tta_map_back",
    "tta_view",
]
