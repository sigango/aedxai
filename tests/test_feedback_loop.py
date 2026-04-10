"""Tests for AED-XAI closed-loop detector refinement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.detector import Detection
from src.evaluator import EvalResult
from src.feedback_loop import FeedbackLoop
from src.xai_methods.base import SaliencyMap


def _make_detection(detection_id: int = 0) -> Detection:
    """Create a simple shared detection for feedback-loop tests."""
    return Detection(
        bbox=[20, 20, 80, 80],
        class_id=0,
        class_name="person",
        confidence=0.35,
        area=3600,
        relative_size="medium",
        detection_id=detection_id,
    )


class _MockDetector:
    """Simple detector stub used to exercise feedback logic."""

    def __init__(self) -> None:
        self.config = {"nms_thresh": 0.45, "conf_thresh": 0.25}
        self.default_nms_thresh = 0.45
        self.default_conf_thresh = 0.25
        self._detections = [_make_detection(0), _make_detection(1)]

    def detect(self, image: np.ndarray, nms_thresh: float | None = None, conf_thresh: float | None = None) -> list[Detection]:
        del image, nms_thresh, conf_thresh
        return list(self._detections)

    def get_model(self) -> object:
        return object()

    def get_target_layer(self) -> None:
        return None

    def compute_scene_complexity(self, detections: list[Detection]) -> str:
        del detections
        return "medium"


@dataclass
class _MockAssessment:
    """Minimal VLM assessment stub."""

    scene_complexity: str = "medium"


class _MockVLMJudge:
    """VLM judge stub that returns a stable scene complexity."""

    def assess_detections(self, image: np.ndarray, detections: list[Detection]) -> list[_MockAssessment]:
        del image, detections
        return [_MockAssessment(scene_complexity="medium")]


class _MockSelector:
    """Selector stub that always chooses GradCAM."""

    def predict(self, detection: Detection, scene_complexity: str, num_detections: int, image: np.ndarray) -> str:
        del detection, scene_complexity, num_detections, image
        return "gradcam"


class _MockExplainer:
    """Explainer stub that returns a fixed centered saliency map."""

    def explain(self, model: object, image: np.ndarray, detection: Detection, target_layer: object | None = None) -> SaliencyMap:
        del model, target_layer
        saliency = np.zeros(image.shape[:2], dtype=np.float32)
        x1, y1, x2, y2 = detection.bbox
        saliency[y1:y2, x1:x2] = 1.0
        return SaliencyMap(
            map=saliency,
            method_name="gradcam",
            computation_time=0.01,
            detection_id=detection.detection_id,
        )


class _MockEvaluator:
    """Evaluator stub that keeps the composite score low to trigger refinement."""

    def __init__(self) -> None:
        self.calls = 0

    def evaluate_all(
        self,
        saliency_map: np.ndarray,
        bbox: list[int],
        model: object,
        image: np.ndarray,
        detection: Detection,
    ) -> EvalResult:
        del saliency_map, bbox, model, image, detection
        self.calls += 1
        return EvalResult(
            pg=0.2,
            ebpg=0.2,
            oa=0.1,
            insertion_auc=0.2,
            deletion_auc=0.1,
            sparsity=0.3,
            composite=0.2,
            computation_time=0.01,
        )


def test_threshold_clamping() -> None:
    """Threshold adjustments should remain within the configured valid range."""
    loop = FeedbackLoop()
    extreme_results = [
        EvalResult(
            pg=0.0,
            ebpg=0.0,
            oa=-0.5,
            insertion_auc=0.0,
            deletion_auc=0.5,
            sparsity=0.0,
            composite=0.0,
            computation_time=1.0,
        )
    ]
    nms = 0.45
    conf = 0.25

    for _ in range(20):
        nms, conf = loop.compute_new_thresholds(extreme_results, nms, conf)
        assert 0.30 <= nms <= 0.70
        assert 0.10 <= conf <= 0.70


def test_feedback_loop_terminates() -> None:
    """The feedback loop should stop within the configured iteration budget."""
    loop = FeedbackLoop()
    detector = _MockDetector()
    result = loop.run(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        image_path="mock.jpg",
        detector=detector,
        vlm_judge=_MockVLMJudge(),
        xai_selector=_MockSelector(),
        xai_methods={"gradcam": _MockExplainer()},
        evaluator=_MockEvaluator(),
    )
    assert len(result.iterations) <= loop.config["max_iterations"]


def test_feedback_improvement_tracking() -> None:
    """Each iteration should store per-detection composite scores."""
    loop = FeedbackLoop()
    result = loop.run(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        image_path="mock.jpg",
        detector=_MockDetector(),
        vlm_judge=_MockVLMJudge(),
        xai_selector=_MockSelector(),
        xai_methods={"gradcam": _MockExplainer()},
        evaluator=_MockEvaluator(),
    )

    for iteration in result.iterations:
        assert iteration.iteration >= 0
        assert 0.0 <= iteration.mean_composite_score <= 1.0
        assert len(iteration.per_detection_scores) == iteration.num_detections
