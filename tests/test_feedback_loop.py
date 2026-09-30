"""Tests for AED-XAI closed-loop detector refinement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

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


class _ConstantExplainer:
    """Explainer stub returning a constant-valued saliency map."""

    def __init__(self, method_name: str, value: float) -> None:
        self.method_name = method_name
        self.value = value

    def explain(self, model: object, image: np.ndarray, detection: Detection, target_layer: object | None = None) -> SaliencyMap:
        del model, target_layer
        saliency = np.full(image.shape[:2], self.value, dtype=np.float32)
        return SaliencyMap(
            map=saliency,
            method_name=self.method_name,
            computation_time=0.01,
            detection_id=detection.detection_id,
        )


class _MaxValueEvaluator:
    """Evaluator stub whose composite equals the saliency map's peak value."""

    def evaluate_all(
        self,
        saliency_map: np.ndarray,
        bbox: list[int],
        model: object,
        image: np.ndarray,
        detection: Detection,
    ) -> EvalResult:
        del bbox, model, image, detection
        composite = float(np.max(np.asarray(saliency_map)))
        return EvalResult(
            pg=composite,
            ebpg=composite,
            oa=composite,
            insertion_auc=composite,
            deletion_auc=0.0,
            sparsity=composite,
            composite=composite,
            computation_time=0.01,
        )


def test_escalation_keeps_better_backup_map() -> None:
    """Low-scoring detections should escalate to the backup and keep the max."""
    loop = FeedbackLoop(
        {
            "threshold_mode": "fixed",
            "threshold": 0.0,
            "max_iterations": 1,
            "escalation": {"enabled": True, "backup_method": "lime", "tau": 0.6},
        }
    )
    result = loop.run(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        image_path="mock.jpg",
        detector=_MockDetector(),
        vlm_judge=_MockVLMJudge(),
        xai_selector=_MockSelector(),
        xai_methods={
            "gradcam": _ConstantExplainer("gradcam", 0.4),
            "lime": _ConstantExplainer("lime", 0.9),
        },
        evaluator=_MaxValueEvaluator(),
    )

    assert result.num_escalations == 2
    assert all(eval_result.composite == pytest.approx(0.9, abs=1e-6) for eval_result in result.final_eval_results)
    assert all(saliency.method_name == "lime" for saliency in result.final_saliency_maps)
    for reasoning in result.final_selector_reasoning.values():
        assert reasoning["escalated"] is True
        assert reasoning["escalation_kept"] == "backup"


def test_escalation_keeps_default_when_backup_is_worse() -> None:
    """Escalation must never replace a map with a lower-scoring backup."""
    loop = FeedbackLoop(
        {
            "threshold_mode": "fixed",
            "threshold": 0.0,
            "max_iterations": 1,
            "escalation": {"enabled": True, "backup_method": "lime", "tau": 0.6},
        }
    )
    result = loop.run(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        image_path="mock.jpg",
        detector=_MockDetector(),
        vlm_judge=_MockVLMJudge(),
        xai_selector=_MockSelector(),
        xai_methods={
            "gradcam": _ConstantExplainer("gradcam", 0.4),
            "lime": _ConstantExplainer("lime", 0.2),
        },
        evaluator=_MaxValueEvaluator(),
    )

    assert result.num_escalations == 2
    assert all(eval_result.composite == pytest.approx(0.4, abs=1e-6) for eval_result in result.final_eval_results)
    assert all(saliency.method_name == "gradcam" for saliency in result.final_saliency_maps)
    for reasoning in result.final_selector_reasoning.values():
        assert reasoning["escalation_kept"] == "default"


@dataclass
class _FPAssessment:
    """VLM assessment stub carrying a false-positive flag."""

    detection_id: int
    is_false_positive: bool
    scene_complexity: str = "medium"


class _FPVLMJudge:
    """VLM judge stub flagging detection 0 as a false positive."""

    def assess_detections(self, image: np.ndarray, detections: list[Detection]) -> list[_FPAssessment]:
        del image
        return [
            _FPAssessment(detection_id=detection.detection_id, is_false_positive=detection.detection_id == 0)
            for detection in detections
        ]


def test_vlm_fp_filter_removes_flagged_detections() -> None:
    """With vlm_fp_filter on, VLM-flagged boxes are excluded before explaining."""
    loop = FeedbackLoop(
        {
            "threshold_mode": "fixed",
            "threshold": 0.0,
            "max_iterations": 1,
            "vlm_fp_filter": True,
        }
    )
    result = loop.run(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        image_path="mock.jpg",
        detector=_MockDetector(),
        vlm_judge=_FPVLMJudge(),
        xai_selector=_MockSelector(),
        xai_methods={"gradcam": _ConstantExplainer("gradcam", 0.9)},
        evaluator=_MaxValueEvaluator(),
    )

    assert [detection.detection_id for detection in result.final_detections] == [1]


class _NamedDetector(_MockDetector):
    """Mock detector with a model_name so per-detector tau can be resolved."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name


def test_escalation_uses_per_detector_tau() -> None:
    """tau_by_detector must override the fallback tau for the matching detector."""
    config = {
        "threshold_mode": "fixed",
        "threshold": 0.0,
        "max_iterations": 1,
        "escalation": {"enabled": True, "backup_method": "lime", "tau": 0.9, "tau_by_detector": {"det-a": 0.3}},
    }
    methods = {"gradcam": _ConstantExplainer("gradcam", 0.4), "lime": _ConstantExplainer("lime", 0.9)}
    loop = FeedbackLoop(config)
    kept = loop.run(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        image_path="mock.jpg",
        detector=_NamedDetector("det-a"),
        vlm_judge=_MockVLMJudge(),
        xai_selector=_MockSelector(),
        xai_methods=methods,
        evaluator=_MaxValueEvaluator(),
    )
    assert kept.num_escalations == 0  # 0.4 >= tau 0.3 for det-a
    escalated = loop.run(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        image_path="mock.jpg",
        detector=_NamedDetector("det-b"),
        vlm_judge=_MockVLMJudge(),
        xai_selector=_MockSelector(),
        xai_methods=methods,
        evaluator=_MaxValueEvaluator(),
    )
    assert escalated.num_escalations == 2  # fallback tau 0.9


def test_fixed_method_bypasses_selector() -> None:
    """fixed_method must be used for every detection regardless of the selector."""
    loop = FeedbackLoop({"threshold_mode": "fixed", "threshold": 0.0, "max_iterations": 1, "fixed_method": "gcame"})
    result = loop.run(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        image_path="mock.jpg",
        detector=_MockDetector(),
        vlm_judge=_MockVLMJudge(),
        xai_selector=_MockSelector(),
        xai_methods={"gcame": _ConstantExplainer("gcame", 0.7), "gradcam": _ConstantExplainer("gradcam", 0.1)},
        evaluator=_MaxValueEvaluator(),
    )
    assert all(saliency.method_name == "gcame" for saliency in result.final_saliency_maps)
    assert all(reasoning["source"] == "fixed" for reasoning in result.final_selector_reasoning.values())


class _SequenceDetector(_MockDetector):
    """Returns distinct boxes whose ids change between calls (like a re-detection)."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def detect(self, image: np.ndarray, nms_thresh: float | None = None, conf_thresh: float | None = None) -> list[Detection]:
        del image, nms_thresh, conf_thresh
        self.calls += 1
        offset = 0 if self.calls == 1 else 5
        boxes = [[10, 10, 40, 40], [60, 60, 90, 90]]
        return [
            Detection(
                bbox=box,
                class_id=0,
                class_name="person",
                confidence=0.5,
                area=900,
                relative_size="medium",
                detection_id=index + offset,
            )
            for index, box in enumerate(boxes)
        ]


class _LowScoreEvaluator(_MaxValueEvaluator):
    """Evaluator whose composites force refinement (left-skewed scores)."""

    def evaluate_all(self, saliency_map, bbox, model, image, detection) -> EvalResult:
        del saliency_map, model, image, detection
        value = 0.1 if bbox[0] == 10 else 0.9
        return EvalResult(pg=value, ebpg=value, oa=value, insertion_auc=value, deletion_auc=0.0, sparsity=value, composite=value, computation_time=0.0)


def test_vlm_fp_filter_persists_across_iterations() -> None:
    """A box flagged at iteration 0 must stay filtered after re-detection with new ids."""
    loop = FeedbackLoop(
        {"threshold_mode": "fixed", "threshold": 1.0, "max_iterations": 2, "min_improvement": -1.0, "vlm_fp_filter": True}
    )
    detector = _SequenceDetector()
    result = loop.run(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        image_path="mock.jpg",
        detector=detector,
        vlm_judge=_FPVLMJudge(),
        xai_selector=_MockSelector(),
        xai_methods={"gradcam": _ConstantExplainer("gradcam", 0.9)},
        evaluator=_LowScoreEvaluator(),
    )
    assert detector.calls == 2
    assert [iteration.num_detections for iteration in result.iterations] == [1, 1]
    assert [detection.bbox for detection in result.final_detections] == [[60, 60, 90, 90]]
