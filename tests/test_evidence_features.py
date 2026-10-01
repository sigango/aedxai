"""Tests for per-detection evidence features and the VLM verification view."""

from __future__ import annotations

import numpy as np
import pytest

from src.evidence import (
    TTA_VIEWS,
    agreement_features,
    candidate_arrays,
    curve_features,
    map_features,
    output_features,
    prenms_features,
    run_tta,
    tta_features,
    tta_map_back,
    tta_view,
)
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


def test_gradient_features_include_head_group_norms() -> None:
    pytest.importorskip("torch")
    from src.evidence import gradient_features
    from toy_models import ToyDetectionModel

    model = ToyDetectionModel().eval()
    features = gradient_features(
        model,
        toy_image(),
        make_detection([16, 16, 48, 48]),
        model.features[0],
        param_groups={"conv0": ("features.0",), "conv2": ("features.2",), "missing": ("no.such.module",)},
    )
    assert features["rg_conv0_l2"] > 0 and features["rg_conv2_l2"] > 0
    assert not any(key.startswith("rg_missing") for key in features)
    # Self-learning loss -log(s): gradient norm = score-gradient norm / s.
    lite = gradient_features(model, toy_image(), make_detection([16, 16, 48, 48]), model.features[0], param_groups={})
    assert not any(key.startswith("rg_") for key in lite)
    assert features["rg_conv0_l2"] == pytest.approx(lite["grad_param_l2"] / lite["grad_score"], rel=1e-4)


def test_prenms_features_count_overlapping_candidates() -> None:
    detection = make_detection([10, 10, 50, 50], class_id=0, confidence=0.8)
    candidates = [
        ([10, 10, 50, 50], 0, 0.8),   # the detection itself
        ([12, 11, 52, 49], 0, 0.6),   # suppressed duplicate
        ([30, 30, 90, 90], 0, 0.5),   # same class, low IoU
        ([11, 10, 50, 51], 2, 0.4),   # other class on the same object
    ]
    features = prenms_features(detection, *candidate_arrays(candidates))
    assert features["count"] == 2.0 and features["score_max"] == pytest.approx(0.8)
    assert features["other_class_count"] == 1.0 and features["other_class_max_score"] == pytest.approx(0.4)
    assert 0 < features["coord_std"] < 0.1 and features["iou_min"] > 0.8
    empty = prenms_features(detection, *candidate_arrays([]))
    assert empty["count"] == 0.0 and empty["score_mean"] == 0.0


@pytest.mark.parametrize("name", TTA_VIEWS)
def test_tta_views_map_boxes_back(name: str) -> None:
    image = toy_image()
    view, params = tta_view(image, name)
    assert view.shape == image.shape and view.dtype == np.uint8
    box = np.array([[16.0, 16.0, 48.0, 48.0]])
    if name == "hflip":
        forward = np.array([[64 - 48.0, 16.0, 64 - 16.0, 48.0]])
    elif name == "shrink":
        forward = box * np.array([params["sx"], params["sy"], params["sx"], params["sy"]]) + np.array(
            [params["ox"], params["oy"], params["ox"], params["oy"]]
        )
    elif name == "shift":
        forward = box + np.array([params["dx"], params["dy"], params["dx"], params["dy"]])
    else:
        forward = box
    np.testing.assert_allclose(tta_map_back(forward, name, params), box, atol=1e-6)


def test_tta_features_stable_and_unstable_detections() -> None:
    detection = make_detection([16, 16, 48, 48], class_id=0, confidence=0.9)
    stable = [
        {"name": view, "boxes": np.array([[16.0, 16.0, 48.0, 48.0]]), "classes": np.array([0]), "scores": np.array([0.85])}
        for view in TTA_VIEWS
    ]
    features = tta_features(detection, stable)
    assert features["match_rate"] == 1.0 and features["score_mean"] == pytest.approx(0.85)
    assert features["class_flip_rate"] == 0.0 and features["iou_min"] == pytest.approx(1.0)
    flipped = [
        {"name": view, "boxes": np.array([[16.0, 16.0, 48.0, 48.0]]), "classes": np.array([2]), "scores": np.array([0.7])}
        for view in TTA_VIEWS
    ]
    features = tta_features(detection, flipped)
    assert features["match_rate"] == 0.0 and features["score_mean"] == 0.0 and features["class_flip_rate"] == 1.0


def test_run_tta_with_toy_wrapper() -> None:
    pytest.importorskip("torch")
    from toy_models import ToyDetectorWrapper

    outputs = run_tta(ToyDetectorWrapper(), toy_image())
    assert [output["name"] for output in outputs] == list(TTA_VIEWS)
    assert all(output["boxes"].shape[1] == 4 for output in outputs)


def test_verification_view_crops() -> None:
    image = toy_image()
    detection = make_detection([16, 16, 48, 48])
    box_view = build_verification_view(image, detection, None, box_context_scale=1.0)
    assert box_view.size == (32, 32)
    saliency = np.zeros((64, 64), dtype=np.float32)
    saliency[0:8, 0:8] = 1.0
    sal_view = build_verification_view(image, detection, saliency, salient_box_multiple=64 / 1024, margin_fraction=0.0)
    assert sal_view.size == (8, 8)
