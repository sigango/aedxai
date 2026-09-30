"""Tests for insertion/deletion curves, batching, the box protocol, and evaluate_evidence."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from src.evaluator import AutoEvaluator
from toy_models import ToyDetectionModel, make_detection, toy_image


def _legacy_insertion_deletion(evaluator, saliency, image, detection, model, steps=10):
    """Reference copy of the pre-refactor per-pixel loop."""
    baseline = evaluator._create_baseline_image(image)
    flat = np.argsort(saliency.reshape(-1))[::-1]
    groups = np.array_split(flat, steps)
    xs = np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)
    curves = []
    for start, source in ((baseline.copy(), image), (image.copy(), baseline)):
        current, scores = start, []
        for step in range(steps + 1):
            scores.append(evaluator._get_detection_confidence(model, current, detection))
            if step < steps:
                for index in groups[step]:
                    y, x = np.unravel_index(int(index), saliency.shape)
                    current[y, x] = source[y, x]
        curves.append(scores)
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return [float(np.clip(trapz(c, xs), 0, 1)) for c in curves], curves


@pytest.fixture(scope="module")
def setup():
    model = ToyDetectionModel().eval()
    image = toy_image()
    detection = make_detection([16, 16, 48, 48])
    saliency = np.random.default_rng(1).random((64, 64)).astype(np.float32)
    return AutoEvaluator(), model, image, detection, saliency


def test_refactor_matches_legacy_loop(setup) -> None:
    evaluator, model, image, detection, saliency = setup
    (ins, dele), curves = _legacy_insertion_deletion(evaluator, saliency, image, detection, model)
    assert evaluator.insertion_deletion(saliency, detection.bbox, model, image, detection) == (ins, dele)
    result = evaluator.insertion_deletion_curves(saliency, model, image, detection)
    assert np.allclose(result["insertion_curve"], curves[0]) and np.allclose(result["deletion_curve"], curves[1])


def test_batched_scoring_matches_unbatched(setup) -> None:
    evaluator, model, image, detection, saliency = setup
    single = evaluator.insertion_deletion_curves(saliency, model, image, detection, batch_size=1)
    batched = evaluator.insertion_deletion_curves(saliency, model, image, detection, batch_size=5)
    assert np.allclose(single["insertion_curve"], batched["insertion_curve"], atol=1e-6)
    assert np.allclose(single["deletion_curve"], batched["deletion_curve"], atol=1e-6)


def test_box_protocol_leaves_outside_pixels_untouched(setup, monkeypatch) -> None:
    evaluator, model, image, detection, saliency = setup
    seen: list[np.ndarray] = []
    original = evaluator._get_detection_confidences

    def _capture(model_, images, detection_, batch_size):
        seen.extend(images)
        return original(model_, images, detection_, batch_size)

    monkeypatch.setattr(evaluator, "_get_detection_confidences", _capture)
    region = (20, 20, 40, 40)
    evaluator.insertion_deletion_curves(saliency, model, image, detection, region=region)
    outside = np.ones(image.shape[:2], dtype=bool)
    outside[20:41, 20:41] = False
    assert seen and all(np.array_equal(step[outside], image[outside]) for step in seen)
    assert np.array_equal(seen[10], image)  # insertion ends at the original image


def test_expanded_box_region_is_clipped() -> None:
    region = AutoEvaluator().expanded_box_region([0, 0, 20, 10], (30, 30), context_scale=2.0)
    assert region == (0, 0, 29, 15)


def test_evaluate_evidence_matches_evaluate_all(setup) -> None:
    evaluator, model, image, detection, saliency = setup
    record = evaluator.evaluate_evidence(saliency, detection, model, image, gt_bbox=[18, 18, 46, 46])
    legacy = evaluator.evaluate_all(saliency_map=saliency, bbox=detection.bbox, model=model, image=image, detection=detection)
    for key in ("pg", "ebpg", "oa", "sparsity", "composite", "insertion_auc", "deletion_auc"):
        assert record[key] == pytest.approx(getattr(legacy, key))
    for key in ("oa_box", "composite_box", "pg_gt", "ebpg_gt", "insertion_curve_box", "eval_time"):
        assert key in record
    assert len(record["insertion_curve"]) == 11
