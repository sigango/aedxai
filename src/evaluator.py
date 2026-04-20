"""Annotation-free explanation evaluation for AED-XAI."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

from .xai_methods.base import (
    SaliencyMap,
    _extract_candidates_from_output,
    _forward_detector,
    _iou_to_target,
)

if TYPE_CHECKING:
    import torch

    from .detector import Detection

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MetricScore:
    """Backward-compatible metric container."""

    name: str
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalResult:
    """Complete annotation-free evaluation result for one explanation."""

    pg: float
    ebpg: float
    oa: float
    insertion_auc: float
    deletion_auc: float
    sparsity: float
    composite: float
    computation_time: float

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access for legacy call sites."""
        return getattr(self, key)

    def as_dict(self) -> dict[str, float]:
        """Convert the result into a flat dictionary."""
        return {
            "pg": float(self.pg),
            "ebpg": float(self.ebpg),
            "oa": float(self.oa),
            "insertion_auc": float(self.insertion_auc),
            "deletion_auc": float(self.deletion_auc),
            "sparsity": float(self.sparsity),
            "composite": float(self.composite),
            "computation_time": float(self.computation_time),
        }


EvaluationResult = EvalResult


def _resolve_path(path: str) -> Path:
    """Resolve a possibly relative config path against the project root."""
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


def _load_eval_config(config_source: str | Mapping[str, Any]) -> dict[str, Any]:
    """Load the evaluation section from YAML or accept a ready mapping."""
    if isinstance(config_source, Mapping):
        raw = dict(config_source)
        if "evaluation" in raw and isinstance(raw["evaluation"], Mapping):
            return dict(raw["evaluation"])
        return raw

    resolved = _resolve_path(str(config_source))
    with resolved.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if "evaluation" in raw and isinstance(raw["evaluation"], Mapping):
        return dict(raw["evaluation"])
    return dict(raw)


def _unwrap_model(model: Any) -> Any:
    """Unwrap helper objects that expose a raw detector model."""
    if hasattr(model, "get_model") and callable(model.get_model):
        return model.get_model()
    return model


def _coerce_saliency_array(saliency_map: np.ndarray | SaliencyMap | Any) -> np.ndarray:
    """Convert supported saliency inputs into a 2D float32 NumPy array."""
    if isinstance(saliency_map, SaliencyMap):
        array = saliency_map.map
    elif hasattr(saliency_map, "map"):
        array = getattr(saliency_map, "map")
    else:
        array = saliency_map

    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"saliency_map must be 2D, got shape {array.shape}")
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


class AutoEvaluator:
    """Annotation-free evaluation of XAI explanations for object detection."""

    def __init__(self, config_path: str | Mapping[str, Any] = "config/eval_config.yaml") -> None:
        """Load metric configurations and composite-score weights."""
        self.config = _load_eval_config(config_path)
        self.metrics_config = dict(self.config.get("metrics", {}))
        _equal_weight = round(1.0 / 3.0, 6)  # 0.333333
        self.composite_weights = dict(
            self.config.get(
                "composite_weights",
                {
                    "ebpg": _equal_weight,
                    "oa": _equal_weight,
                    "sparsity": _equal_weight,
                },
            )
        )
        # Backward compat: legacy configs with "pg" weight are treated as "ebpg".
        if "ebpg" not in self.composite_weights and "pg" in self.composite_weights:
            self.composite_weights["ebpg"] = self.composite_weights.pop("pg")

        self.over_all_config = dict(self.metrics_config.get("over_all", {}))
        self.sparsity_config = dict(self.metrics_config.get("sparsity", {}))

    def pointing_game(self, saliency_map: np.ndarray, bbox: list[int]) -> float:
        """Return 1.0 if the peak saliency lies inside the pseudo-ground-truth box."""
        saliency = _coerce_saliency_array(saliency_map)
        if saliency.size == 0 or float(saliency.max()) <= 0.0:
            return 0.0

        max_y, max_x = np.unravel_index(int(np.argmax(saliency)), saliency.shape)
        x1, y1, x2, y2 = [int(value) for value in bbox]
        return 1.0 if x1 <= max_x <= x2 and y1 <= max_y <= y2 else 0.0

    def energy_based_pg(self, saliency_map: np.ndarray, bbox: list[int]) -> float:
        """Return the fraction of total saliency energy that falls inside the box."""
        saliency = _coerce_saliency_array(saliency_map)
        x1, y1, x2, y2 = [int(value) for value in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(saliency.shape[1] - 1, x2)
        y2 = min(saliency.shape[0] - 1, y2)

        if x2 < x1 or y2 < y1:
            return 0.0

        energy_total = float(saliency.sum())
        if energy_total < 1e-8:
            return 0.0

        energy_inside = float(saliency[y1 : y2 + 1, x1 : x2 + 1].sum())
        return float(np.clip(energy_inside / energy_total, 0.0, 1.0))

    def insertion_deletion(
        self,
        saliency_map: np.ndarray,
        bbox: list[int],
        model: "torch.nn.Module" | Any,
        image: np.ndarray,
        detection: "Detection",
    ) -> tuple[float, float]:
        """Compute insertion and deletion AUCs via repeated detector forward passes."""
        del bbox

        saliency = _coerce_saliency_array(saliency_map)
        image = np.asarray(image, dtype=np.uint8)
        if image.ndim != 3 or image.shape[:2] != saliency.shape:
            raise ValueError(
                f"image shape {image.shape} is incompatible with saliency map shape {saliency.shape}"
            )

        steps = max(1, int(self.over_all_config.get("num_steps", 10)))
        baseline = self._create_baseline_image(image)
        flat_indices = np.argsort(saliency.reshape(-1))[::-1]
        pixel_groups = np.array_split(flat_indices, steps)
        x_coords = np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)

        insertion_scores: list[float] = []
        current_insertion = baseline.copy()
        for step_index in range(steps + 1):
            insertion_scores.append(self._get_detection_confidence(model, current_insertion, detection))
            if step_index >= steps:
                continue
            for flat_index in pixel_groups[step_index]:
                y_coord, x_coord = np.unravel_index(int(flat_index), saliency.shape)
                current_insertion[y_coord, x_coord] = image[y_coord, x_coord]

        deletion_scores: list[float] = []
        current_deletion = image.copy()
        for step_index in range(steps + 1):
            deletion_scores.append(self._get_detection_confidence(model, current_deletion, detection))
            if step_index >= steps:
                continue
            for flat_index in pixel_groups[step_index]:
                y_coord, x_coord = np.unravel_index(int(flat_index), saliency.shape)
                current_deletion[y_coord, x_coord] = baseline[y_coord, x_coord]

        trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        insertion_auc = float(np.clip(trapz(insertion_scores, x_coords), 0.0, 1.0))
        deletion_auc = float(np.clip(trapz(deletion_scores, x_coords), 0.0, 1.0))
        return insertion_auc, deletion_auc

    def _get_detection_confidence(
        self,
        model: "torch.nn.Module" | Any,
        image: np.ndarray,
        original_detection: "Detection",
    ) -> float:
        """Find the confidence of the best-matching detection in a perturbed image."""
        raw_model = _unwrap_model(model)
        output, _, meta = _forward_detector(raw_model, image, require_grad=False)
        boxes, scores, class_ids = _extract_candidates_from_output(output, model=raw_model, meta=meta)
        if boxes.numel() == 0:
            return 0.0

        import torch

        target_box = torch.as_tensor(original_detection.bbox, dtype=boxes.dtype, device=boxes.device)
        ious = _iou_to_target(boxes, target_box)
        same_class = class_ids == int(original_detection.class_id)
        valid = same_class & (ious > 0.5)
        if not bool(valid.any().item()):
            return 0.0

        valid_indices = torch.arange(boxes.shape[0], device=boxes.device)[valid]
        ranking = ious[valid_indices] + (scores[valid_indices] * 1e-3)
        best_index = valid_indices[torch.argmax(ranking)]
        return float(scores[best_index].detach().cpu().item())

    def sparsity_gini(self, saliency_map: np.ndarray) -> float:
        """Compute saliency sparsity via the Gini coefficient."""
        saliency = _coerce_saliency_array(saliency_map)
        values = np.sort(saliency.reshape(-1).astype(np.float64))
        if values.size == 0:
            return 0.0

        total = float(values.sum())
        if total <= 1e-12:
            return 0.0

        n = values.size
        index = np.arange(1, n + 1, dtype=np.float64)
        gini = (2.0 * np.sum(index * values) / (n * total)) - ((n + 1.0) / n)
        return float(np.clip(gini, 0.0, 1.0))

    def composite_score(
        self,
        eval_result_partial: Mapping[str, float] | EvalResult,
        weights: Mapping[str, float] | None = None,
    ) -> float:
        """Compute the weighted AED-XAI composite score in the stable [0, 1] range."""
        source = eval_result_partial.as_dict() if isinstance(eval_result_partial, EvalResult) else eval_result_partial
        active_weights = dict(weights) if weights is not None else self.composite_weights
        if "ebpg" not in active_weights and "pg" in active_weights:
            active_weights = {**active_weights, "ebpg": active_weights["pg"]}

        # Plausibility component: EBPG (continuous, energy-based) instead of PG.
        # Falls back to "pg" key only if "ebpg" is entirely absent, for compat with
        # legacy call sites that still pass {"pg": ...}.
        if "ebpg" in source:
            plausibility = float(source.get("ebpg", 0.0))
        else:
            plausibility = float(source.get("pg", 0.0))
        plausibility = float(np.clip(plausibility, 0.0, 1.0))

        oa = float(source.get("oa", 0.0))
        # OA = insertion_AUC - deletion_AUC ∈ [-1, 1]. Linear map to [0, 1]: (oa + 1) / 2.
        oa_norm = float(np.clip((oa + 1.0) / 2.0, 0.0, 1.0))
        sparsity = float(np.clip(float(source.get("sparsity", 0.0)), 0.0, 1.0))
        equal_weight = round(1.0 / 3.0, 6)

        composite = (
            float(active_weights.get("ebpg", equal_weight)) * plausibility
            + float(active_weights.get("oa", equal_weight)) * oa_norm
            + float(active_weights.get("sparsity", equal_weight)) * sparsity
        )
        return float(np.clip(composite, 0.0, 1.0))

    def evaluate_all(
        self,
        saliency_map: np.ndarray | SaliencyMap | Any = None,
        bbox: list[int] | None = None,
        model: "torch.nn.Module" | Any | None = None,
        image: np.ndarray | None = None,
        detection: "Detection" | None = None,
        **kwargs: Any,
    ) -> EvalResult:
        """Compute all configured metrics for one explanation and return an EvalResult."""
        t_start = time.time()

        if saliency_map is None and "saliency" in kwargs:
            saliency_map = kwargs["saliency"]
        if saliency_map is None:
            raise ValueError("saliency_map is required")
        if bbox is None:
            raise ValueError("bbox is required")

        saliency = _coerce_saliency_array(saliency_map)
        pg = self.pointing_game(saliency, bbox) if self.metrics_config.get("pointing_game", {}).get("enabled", True) else 0.0
        ebpg = (
            self.energy_based_pg(saliency, bbox)
            if self.metrics_config.get("energy_based_pg", {}).get("enabled", True)
            else 0.0
        )

        insertion_auc = 0.0
        deletion_auc = 0.0
        oa = 0.0
        if self.metrics_config.get("over_all", {}).get("enabled", True):
            if model is None or image is None or detection is None:
                raise ValueError("model, image, and detection are required for insertion/deletion metrics")
            insertion_auc, deletion_auc = self.insertion_deletion(
                saliency_map=saliency,
                bbox=bbox,
                model=model,
                image=image,
                detection=detection,
            )
            oa = float(insertion_auc - deletion_auc)

        sparsity = self.sparsity_gini(saliency) if self.metrics_config.get("sparsity", {}).get("enabled", True) else 0.0
        computation_time = float(time.time() - t_start)
        composite = self.composite_score({"ebpg": ebpg, "oa": oa, "sparsity": sparsity})

        result = EvalResult(
            pg=float(pg),
            ebpg=float(ebpg),
            oa=float(oa),
            insertion_auc=float(insertion_auc),
            deletion_auc=float(deletion_auc),
            sparsity=float(sparsity),
            composite=float(composite),
            computation_time=computation_time,
        )
        logger.info(
            "Eval result | pg=%.4f ebpg=%.4f oa=%.4f ins=%.4f del=%.4f sparsity=%.4f composite=%.4f time=%.4fs",
            result.pg,
            result.ebpg,
            result.oa,
            result.insertion_auc,
            result.deletion_auc,
            result.sparsity,
            result.composite,
            result.computation_time,
        )
        return result

    def evaluate(
        self,
        image: np.ndarray,
        detection: "Detection",
        saliency_map: SaliencyMap | np.ndarray,
        model: "torch.nn.Module" | Any,
    ) -> EvalResult:
        """Compatibility wrapper that forwards to evaluate_all."""
        bbox = detection.bbox if hasattr(detection, "bbox") else None
        return self.evaluate_all(saliency_map=saliency_map, bbox=bbox, model=model, image=image, detection=detection)

    def _create_baseline_image(self, image: np.ndarray) -> np.ndarray:
        """Create the perturbation baseline image from the evaluator config."""
        method = str(self.over_all_config.get("baseline_method", "blur")).strip().lower()
        if method == "blur":
            kernel_size = int(self.over_all_config.get("blur_kernel_size", 51))
            kernel_size = max(1, kernel_size)
            if kernel_size % 2 == 0:
                kernel_size += 1
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        if method == "mean":
            mean_color = image.mean(axis=(0, 1)).astype(np.uint8)
            return np.broadcast_to(mean_color, image.shape).copy()
        if method == "black":
            return np.zeros_like(image)
        raise ValueError(f"Unsupported baseline method: {method}")


ExplanationEvaluator = AutoEvaluator

__all__ = [
    "AutoEvaluator",
    "EvalResult",
    "EvaluationResult",
    "ExplanationEvaluator",
    "MetricScore",
]
