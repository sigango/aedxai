"""Functional tests for the AED-XAI detector wrapper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.detector import Detection, DetectorWrapper, compute_scene_complexity
from src.utils import COCO_CLASSES, load_image


@pytest.fixture(scope="module")
def fasterrcnn_detector() -> DetectorWrapper:
    """Load Faster R-CNN once for the detector test module."""
    detector = DetectorWrapper("fasterrcnn_resnet50_fpn_v2")
    detector.load_model()
    return detector


def _load_detector_test_config() -> dict:
    """Load the detector YAML used by the wrapper and scene-complexity tests."""
    config_path = Path(__file__).resolve().parents[1] / "config" / "detector_config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _mock_detection(detection_id: int, bbox: list[int], confidence: float = 0.9) -> Detection:
    """Create a deterministic detection for scene-complexity tests."""
    area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
    return Detection(
        bbox=bbox,
        class_id=0,
        class_name="person",
        confidence=confidence,
        area=area,
        relative_size="small",
        detection_id=detection_id,
    )


def test_yolox_loads() -> None:
    """YOLOX-S loads without error when the optional dependency is installed."""
    if importlib.util.find_spec("yolox") is None:
        pytest.skip("YOLOX is not installed in this environment.")

    det = DetectorWrapper("yolox-s")
    det.load_model()
    assert det.model is not None


def test_fasterrcnn_loads(fasterrcnn_detector: DetectorWrapper) -> None:
    """Faster R-CNN loads without error."""
    assert fasterrcnn_detector.model is not None


def test_detect_returns_detections(sample_image: Path, fasterrcnn_detector: DetectorWrapper) -> None:
    """Detection on a real image returns at least one detection."""
    image = load_image(str(sample_image))
    results = fasterrcnn_detector.detect(image)
    assert len(results) > 0


def test_detection_fields(sample_image: Path, fasterrcnn_detector: DetectorWrapper) -> None:
    """All Detection fields are correctly typed and within valid ranges."""
    results = fasterrcnn_detector.detect(load_image(str(sample_image)))

    for detection in results:
        assert len(detection.bbox) == 4
        assert all(isinstance(coordinate, int) for coordinate in detection.bbox)
        assert detection.bbox[0] < detection.bbox[2]
        assert detection.bbox[1] < detection.bbox[3]
        assert 0 <= detection.class_id <= 79
        assert detection.class_name in COCO_CLASSES
        assert 0.0 <= detection.confidence <= 1.0
        assert detection.area > 0
        assert detection.relative_size in ("small", "medium", "large")


def test_target_layer_exists(fasterrcnn_detector: DetectorWrapper) -> None:
    """get_target_layer() returns a valid nn.Module."""
    layer = fasterrcnn_detector.get_target_layer()
    assert isinstance(layer, torch.nn.Module)


def test_visualization(sample_image: Path, fasterrcnn_detector: DetectorWrapper) -> None:
    """visualize() returns a valid RGB numpy image."""
    image = load_image(str(sample_image))
    results = fasterrcnn_detector.detect(image)
    vis = fasterrcnn_detector.visualize(image, results)
    assert vis.shape == image.shape
    assert vis.dtype == np.uint8


def test_scene_complexity() -> None:
    """Scene complexity correctly classifies scenes with few and many boxes."""
    config = _load_detector_test_config()

    few = [
        _mock_detection(0, [0, 0, 10, 10], 0.9),
        _mock_detection(1, [20, 20, 30, 30], 0.9),
    ]
    many = [
        _mock_detection(index, [index * 20, 0, index * 20 + 10, 10], 0.9)
        for index in range(15)
    ]

    assert compute_scene_complexity(few, config) == "low"
    assert compute_scene_complexity(many, config) == "high"


def test_scene_complexity_with_overlap() -> None:
    """High overlap should bump a low-count scene above the base complexity."""
    config = _load_detector_test_config()
    overlapping = [
        _mock_detection(0, [10, 10, 100, 100], 0.9),
        _mock_detection(1, [15, 15, 105, 105], 0.9),
        _mock_detection(2, [12, 12, 102, 102], 0.9),
    ]

    result = compute_scene_complexity(overlapping, config)
    assert result in ("medium", "high")


def test_nms_threshold_effect(sample_image: Path, fasterrcnn_detector: DetectorWrapper) -> None:
    """Lower NMS threshold produces fewer or equal detections."""
    image = load_image(str(sample_image))
    loose = fasterrcnn_detector.detect(image, nms_thresh=0.7)
    strict = fasterrcnn_detector.detect(image, nms_thresh=0.3)
    assert len(strict) <= len(loose)
