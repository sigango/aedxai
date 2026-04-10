"""Closed-loop refinement of detector thresholds based on XAI quality."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import yaml

if TYPE_CHECKING:
    from .detector import Detection, DetectorWrapper
    from .evaluator import AutoEvaluator, EvalResult
    from .vlm_judge import VLMJudge
    from .xai_methods import XAIExplainer
    from .xai_methods.base import SaliencyMap
    from .xai_selector import XAISelector

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FeedbackIteration:
    """Record of one feedback iteration."""

    iteration: int
    nms_thresh: float
    conf_thresh: float
    num_detections: int
    mean_composite_score: float
    per_detection_scores: list[float]


@dataclass(slots=True)
class FeedbackResult:
    """Complete result of the feedback loop for one image."""

    image_path: str
    iterations: list[FeedbackIteration]
    final_detections: list["Detection"]
    final_saliency_maps: list["SaliencyMap"]
    final_eval_results: list["EvalResult"]
    converged: bool
    total_time: float


FeedbackOutcome = FeedbackResult


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


def _load_feedback_config(config_source: str | Mapping[str, Any]) -> dict[str, Any]:
    """Load the feedback subsection from YAML or accept an already parsed mapping."""
    if isinstance(config_source, Mapping):
        raw = dict(config_source)
        if "evaluation" in raw and isinstance(raw["evaluation"], Mapping):
            raw = dict(raw["evaluation"])
        if "feedback" in raw and isinstance(raw["feedback"], Mapping):
            return dict(raw["feedback"])
        return raw

    resolved = _resolve_path(str(config_source))
    with resolved.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    evaluation = dict(raw.get("evaluation", {}))
    return dict(evaluation.get("feedback", {}))


class FeedbackLoop:
    """Closed-loop detector refinement based on XAI quality scores."""

    def __init__(self, config_path: str | Mapping[str, Any] = "config/eval_config.yaml") -> None:
        """Load feedback config: threshold, max_iterations, steps, and clamp ranges."""
        self.config = _load_feedback_config(config_path)
        self.threshold = float(self.config.get("threshold", 0.5))
        self.max_iterations = int(self.config.get("max_iterations", 3))
        self.conf_thresh_step = float(self.config.get("conf_thresh_step", 0.05))
        self.nms_thresh_step = float(self.config.get("nms_thresh_step", 0.05))
        self.conf_thresh_range = tuple(float(value) for value in self.config.get("conf_thresh_range", [0.10, 0.70]))
        self.nms_thresh_range = tuple(float(value) for value in self.config.get("nms_thresh_range", [0.30, 0.70]))
        self.min_improvement = float(self.config.get("min_improvement", 0.01))

    def should_refine(self, mean_composite: float) -> bool:
        """Return True when the mean composite score is below the refinement threshold."""
        return float(mean_composite) < self.threshold

    def compute_new_thresholds(
        self,
        eval_results: list["EvalResult"],
        current_nms: float,
        current_conf: float,
    ) -> tuple[float, float]:
        """Adjust detector thresholds based on the failure mode of current explanations."""
        if not eval_results:
            new_nms = max(self.nms_thresh_range[0], current_nms - self.nms_thresh_step)
            new_conf = min(self.conf_thresh_range[1], current_conf + self.conf_thresh_step)
            return float(new_nms), float(new_conf)

        mean_pg = float(np.mean([result.pg for result in eval_results]))
        mean_oa = float(np.mean([result.oa for result in eval_results]))

        new_nms = float(current_nms)
        new_conf = float(current_conf)

        localization_low = mean_pg < 0.5
        faithfulness_low = mean_oa < 0.3

        if localization_low:
            new_nms -= self.nms_thresh_step
        if faithfulness_low:
            new_conf += self.conf_thresh_step
        if not localization_low and not faithfulness_low:
            new_conf += self.conf_thresh_step * 0.5

        new_nms = float(np.clip(new_nms, self.nms_thresh_range[0], self.nms_thresh_range[1]))
        new_conf = float(np.clip(new_conf, self.conf_thresh_range[0], self.conf_thresh_range[1]))
        return new_nms, new_conf

    def run(
        self,
        image: np.ndarray,
        image_path: str,
        detector: "DetectorWrapper",
        vlm_judge: "VLMJudge",
        xai_selector: "XAISelector",
        xai_methods: dict[str, "XAIExplainer"],
        evaluator: "AutoEvaluator",
    ) -> FeedbackResult:
        """Run the AED-XAI closed loop on one image until convergence or stopping."""
        t_start = time.time()

        nms_thresh = float(getattr(detector, "default_nms_thresh", detector.config.get("nms_thresh", 0.45)))
        conf_thresh = float(getattr(detector, "default_conf_thresh", detector.config.get("conf_thresh", 0.25)))
        iterations: list[FeedbackIteration] = []

        assessments: list[Any] | None = None
        scene_complexity = "medium"
        prev_composite = -1.0

        last_detections: list["Detection"] = []
        last_saliency_maps: list["SaliencyMap"] = []
        last_eval_results: list["EvalResult"] = []

        for iteration_index in range(self.max_iterations):
            detections = detector.detect(image, nms_thresh=nms_thresh, conf_thresh=conf_thresh)
            last_detections = detections
            if not detections:
                logger.info("Feedback loop ended early: no detections at iteration %d", iteration_index)
                return FeedbackResult(
                    image_path=image_path,
                    iterations=iterations,
                    final_detections=[],
                    final_saliency_maps=[],
                    final_eval_results=[],
                    converged=False,
                    total_time=float(time.time() - t_start),
                )

            if iteration_index == 0:
                assessments = vlm_judge.assess_detections(image, detections)
                if assessments:
                    scene_complexity = str(getattr(assessments[0], "scene_complexity", "medium"))
                else:
                    scene_complexity = (
                        detector.compute_scene_complexity(detections)
                        if hasattr(detector, "compute_scene_complexity")
                        else "medium"
                    )

            selected_methods = {
                detection.detection_id: xai_selector.predict(
                    detection=detection,
                    scene_complexity=scene_complexity,
                    num_detections=len(detections),
                    image=image,
                )
                for detection in detections
            }

            needs_target_layer = any(
                selected_methods[detection.detection_id] in {"gradcam", "gcame"} for detection in detections
            )
            target_layer = detector.get_target_layer() if needs_target_layer else None
            model = detector.get_model()

            saliency_maps: list["SaliencyMap"] = []
            for detection in detections:
                method_name = selected_methods[detection.detection_id]
                explainer = xai_methods[method_name]
                saliency = explainer.explain(
                    model=model,
                    image=image,
                    detection=detection,
                    target_layer=target_layer if method_name in {"gradcam", "gcame"} else None,
                )
                saliency_maps.append(saliency)
            last_saliency_maps = saliency_maps

            eval_results = [
                evaluator.evaluate_all(
                    saliency_map=saliency.map,
                    bbox=detection.bbox,
                    model=model,
                    image=image,
                    detection=detection,
                )
                for detection, saliency in zip(detections, saliency_maps)
            ]
            last_eval_results = eval_results

            mean_composite = float(np.mean([result.composite for result in eval_results])) if eval_results else 0.0
            per_detection_scores = [float(result.composite) for result in eval_results]

            iterations.append(
                FeedbackIteration(
                    iteration=iteration_index,
                    nms_thresh=float(nms_thresh),
                    conf_thresh=float(conf_thresh),
                    num_detections=len(detections),
                    mean_composite_score=mean_composite,
                    per_detection_scores=per_detection_scores,
                )
            )

            logger.info(
                "Iteration %d | nms=%.3f conf=%.3f n_det=%d composite=%.4f",
                iteration_index,
                nms_thresh,
                conf_thresh,
                len(detections),
                mean_composite,
            )

            if not self.should_refine(mean_composite):
                return FeedbackResult(
                    image_path=image_path,
                    iterations=iterations,
                    final_detections=detections,
                    final_saliency_maps=saliency_maps,
                    final_eval_results=eval_results,
                    converged=True,
                    total_time=float(time.time() - t_start),
                )

            improvement = mean_composite - prev_composite if prev_composite >= 0.0 else float("inf")
            if iteration_index > 0 and improvement < self.min_improvement:
                logger.info(
                    "Early stopping: improvement %.4f < min_improvement %.4f",
                    improvement,
                    self.min_improvement,
                )
                return FeedbackResult(
                    image_path=image_path,
                    iterations=iterations,
                    final_detections=detections,
                    final_saliency_maps=saliency_maps,
                    final_eval_results=eval_results,
                    converged=False,
                    total_time=float(time.time() - t_start),
                )

            prev_composite = mean_composite
            nms_thresh, conf_thresh = self.compute_new_thresholds(
                eval_results=eval_results,
                current_nms=nms_thresh,
                current_conf=conf_thresh,
            )

        return FeedbackResult(
            image_path=image_path,
            iterations=iterations,
            final_detections=last_detections,
            final_saliency_maps=last_saliency_maps,
            final_eval_results=last_eval_results,
            converged=False,
            total_time=float(time.time() - t_start),
        )


__all__ = ["FeedbackIteration", "FeedbackLoop", "FeedbackOutcome", "FeedbackResult"]
