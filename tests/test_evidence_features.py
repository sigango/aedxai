"""Tests for per-detection evidence features and the VLM verification view."""

from __future__ import annotations

import numpy as np
import pytest

from src.evidence import agreement_features, curve_features, map_features, output_features
from src.vlm_judge import build_verification_view
from toy_models import make_detection, toy_image


def test_map_features_for_box_concentrated_map() -> None:
    saliency = np.zeros((64, 64), dtype=np.float32)
    saliency[20:40, 20:40] = 1.0
    features = map_features(saliency, [20, 20, 40, 40])
    assert features["mass_in_box"] == pytest.approx(1.0)
    assert features["peak_in_box"] == 1.0
    assert features["topk_in_box"] == pytest.approx(1.0)
    assert features["com_offset"] < 0.05


def test_map_features_for_context_map() -> None:
    saliency = np.zeros((64, 64), dtype=np.float32)
    saliency[0:5, 0:5] = 1.0
    features = map_features(saliency, [40, 40, 60, 60])
    assert features["mass_in_box"] == 0.0 and features["mass_outside_context"] == pytest.approx(1.0)


def test_curve_features_normalized_by_reference() -> None:
    features = curve_features([0.0, 0.8, 0.9], [0.9, 0.1, 0.0])
    assert features["ref_score"] == pytest.approx(0.9)
    assert features["del_first_drop"] == pytest.approx((0.9 - 0.1) / 0.9)
    assert features["del_half_step"] == pytest.approx(0.5)
    assert features["del_monotone"] == 1.0


def test_agreement_identical_and_disjoint() -> None:
    rng = np.random.default_rng(0)
    a = rng.random((64, 64)).astype(np.float32)
    same = agreement_features(a, a.copy(), [10, 10, 40, 40])
    assert same["rank_corr"] == pytest.approx(1.0) and same["top10_iou"] == pytest.approx(1.0)
    left = np.zeros((64, 64), dtype=np.float32)
    left[:, :20] = np.linspace(0, 1, 20)
    right = left[:, ::-1].copy()
    assert agreement_features(left, right, [0, 0, 64, 64])["top10_iou"] == 0.0


def test_output_features() -> None:
    image = toy_image()
    first = make_detection([16, 16, 48, 48], confidence=0.9, detection_id=0)
    second = make_detection([20, 20, 50, 50], confidence=0.5, detection_id=1)
    features = output_features(second, [first, second], image)
    assert features["max_iou_same_class"] > 0.5
    assert features["confidence_rank"] == 1.0 and features["num_detections"] == 2.0


def test_gradient_features_with_toy_model() -> None:
    pytest.importorskip("torch")
    from src.evidence import gradient_features
    from toy_models import ToyDetectionModel

    model = ToyDetectionModel().eval()
    features = gradient_features(model, toy_image(), make_detection([16, 16, 48, 48]), model.features[0])
    assert features["grad_l2"] > 0 and features["grad_param_l2"] > 0 and 0 < features["grad_score"] < 1


def test_verification_view_crops() -> None:
    image = toy_image()
    detection = make_detection([16, 16, 48, 48])
    box_view = build_verification_view(image, detection, None, box_context_scale=1.0)
    assert box_view.size == (32, 32)
    saliency = np.zeros((64, 64), dtype=np.float32)
    saliency[0:8, 0:8] = 1.0
    sal_view = build_verification_view(image, detection, saliency, salient_box_multiple=64 / 1024, margin_fraction=0.0)
    assert sal_view.size == (8, 8)
