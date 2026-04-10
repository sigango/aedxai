"""Tests for the AED-XAI adaptive selector."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.detector import Detection
from src.xai_selector import METHOD_NAMES, XAISelector, XAISelectorMLP


def _make_detection(confidence: float, relative_size: str) -> Detection:
    """Create a small helper detection for selector tests."""
    return Detection(
        bbox=[0, 0, 100, 100],
        class_id=0,
        class_name="person",
        confidence=confidence,
        area=10000,
        relative_size=relative_size,
        detection_id=0,
    )


def test_rule_based_returns_valid_method() -> None:
    """Fallback always returns a valid method name."""
    selector = XAISelector()
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    for confidence in [0.1, 0.5, 0.9]:
        for complexity in ["low", "medium", "high"]:
            for size in ["small", "medium", "large"]:
                detection = _make_detection(confidence, size)
                result = selector.predict(detection, complexity, 5, image)
                assert result in METHOD_NAMES


def test_rule_based_high_conf_simple() -> None:
    """High confidence + low complexity should favor GCAME."""
    selector = XAISelector()
    detection = _make_detection(0.9, "medium")
    result = selector.rule_based_fallback(detection, "low")
    assert result == "gcame"


def test_rule_based_low_conf_complex() -> None:
    """Low confidence + high complexity should favor D-CLOSE."""
    selector = XAISelector()
    detection = _make_detection(0.3, "medium")
    result = selector.rule_based_fallback(detection, "high")
    assert result == "dclose"


def test_mlp_architecture() -> None:
    """MLP has the requested input and output dimensions."""
    mlp = XAISelectorMLP()
    dummy_input = torch.randn(1, 7)
    output = mlp(dummy_input)
    assert output.shape == (1, 4)


def test_mlp_training_on_synthetic_data(tmp_path: Path) -> None:
    """MLP should beat the random baseline on patterned synthetic data."""
    rng = np.random.default_rng(0)
    rows = []

    for _ in range(600):
        class_id = int(rng.integers(0, 80))
        confidence = float(rng.uniform(0.0, 1.0))
        relative_size_encoded = int(rng.integers(0, 3))
        scene_complexity_encoded = int(rng.integers(0, 3))
        num_detections = int(rng.integers(1, 20))
        bbox_aspect_ratio = float(rng.uniform(0.5, 2.5))
        image_entropy = float(rng.uniform(1.0, 7.5))

        if confidence >= 0.75 and scene_complexity_encoded == 0:
            label = 1
        elif confidence < 0.35 or scene_complexity_encoded == 2:
            label = 2
        elif relative_size_encoded == 2 and scene_complexity_encoded == 1:
            label = 3
        else:
            label = 0

        rows.append(
            {
                "class_id": class_id,
                "confidence": confidence,
                "relative_size_encoded": relative_size_encoded,
                "scene_complexity_encoded": scene_complexity_encoded,
                "num_detections": num_detections,
                "bbox_aspect_ratio": bbox_aspect_ratio,
                "image_entropy": image_entropy,
                "best_method_label": label,
            }
        )

    dataframe = pd.DataFrame(rows)
    csv_path = tmp_path / "synthetic_selector_data.csv"
    dataframe.to_csv(csv_path, index=False)

    selector = XAISelector(model_path=str(tmp_path / "selector.pth"))
    metrics = selector.train(str(csv_path), epochs=40, batch_size=64, lr=0.001)
    assert metrics["test_acc"] > 0.40


def test_prediction_speed() -> None:
    """Rule-based single prediction should take under 1 ms."""
    selector = XAISelector()
    detection = _make_detection(0.9, "medium")
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    start = time.time()
    for _ in range(1000):
        selector.predict(detection, "low", 5, image)
    elapsed = (time.time() - start) / 1000.0
    assert elapsed < 0.001


def test_probabilities_sum_to_one() -> None:
    """predict_with_probabilities returns a valid distribution."""
    selector = XAISelector()
    detection = _make_detection(0.9, "medium")
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    probabilities = selector.predict_with_probabilities(detection, "low", 5, image)

    assert abs(sum(probabilities.values()) - 1.0) < 1e-5
    assert all(value >= 0.0 for value in probabilities.values())
