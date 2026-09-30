"""Tests for model-free control explanations and explainer variants."""

from __future__ import annotations

import numpy as np

from src.detector import Detection
from src.evaluator import AutoEvaluator
from src.xai_methods import EXPLAINER_VARIANTS, GCAMEExplainer, GradCAMExplainer, get_explainer, shuffle_saliency


def _detection() -> Detection:
    return Detection(bbox=[10, 10, 30, 40], class_id=0, class_name="person", confidence=0.9, area=600, relative_size="medium", detection_id=3)


def test_controls_return_normalized_maps() -> None:
    image = np.zeros((50, 60, 3), dtype=np.uint8)
    for name in ("bbox_prior", "center_prior", "random"):
        saliency = get_explainer(name, {}).explain(None, image, _detection())
        assert saliency.map.shape == (50, 60)
        assert 0.0 <= float(saliency.map.min()) and float(saliency.map.max()) <= 1.0
        assert saliency.detection_id == 3 and saliency.is_fallback is False


def test_random_control_is_deterministic_per_detection() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    first = get_explainer("random", {"seed": 1}).explain(None, image, _detection()).map
    second = get_explainer("random", {"seed": 1}).explain(None, image, _detection()).map
    assert np.array_equal(first, second)


def test_bbox_prior_peaks_inside_box() -> None:
    image = np.zeros((50, 60, 3), dtype=np.uint8)
    saliency = get_explainer("bbox_prior", {}).explain(None, image, _detection()).map
    assert AutoEvaluator().pointing_game(saliency, [10, 10, 30, 40]) == 1.0


def test_shuffle_preserves_gini_but_moves_mass() -> None:
    image = np.zeros((50, 60, 3), dtype=np.uint8)
    saliency = get_explainer("bbox_prior", {}).explain(None, image, _detection()).map
    shuffled = shuffle_saliency(saliency, seed=0)
    evaluator = AutoEvaluator()
    assert np.isclose(evaluator.sparsity_gini(saliency), evaluator.sparsity_gini(shuffled))
    assert evaluator.energy_based_pg(shuffled, [10, 10, 30, 40]) < evaluator.energy_based_pg(saliency, [10, 10, 30, 40])


def test_raw_variants_remove_bbox_conditioning() -> None:
    assert set(EXPLAINER_VARIANTS) == {"gcame_raw", "gradcam_raw"}
    gcame_raw = get_explainer("gcame_raw", {"use_bbox_center_weighting": True})
    gradcam_raw = get_explainer("gradcam_raw", {"renormalize_in_bbox": True})
    assert isinstance(gcame_raw, GCAMEExplainer) and gcame_raw.config["use_bbox_center_weighting"] is False
    assert isinstance(gradcam_raw, GradCAMExplainer) and gradcam_raw.config["renormalize_in_bbox"] is False
