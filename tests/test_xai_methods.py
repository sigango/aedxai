"""Tests for AED-XAI explanation methods and their shared utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from src.detector import Detection, DetectorWrapper
from src.utils import load_image
from src.xai_methods import get_explainer
from src.xai_methods.base import XAIExplainer
from src.xai_methods.dclose import DCLOSEExplainer
from src.xai_methods.gcame import GCAMEExplainer
from src.xai_methods.gradcam import GradCAMExplainer
from src.xai_methods.lime_det import LIMEExplainer


def _load_method_config(method_name: str) -> dict:
    """Load one method configuration from xai_config.yaml."""
    config_path = Path(__file__).resolve().parents[1] / "config" / "xai_config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return dict(config["xai"]["methods"][method_name])


class ToyDetectionModel(torch.nn.Module):
    """Small differentiable detector-like model for fast explainer tests."""

    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
        )

        with torch.no_grad():
            for module in self.features:
                if isinstance(module, torch.nn.Conv2d):
                    module.weight.fill_(1.0 / module.weight.numel())

    def forward(self, images):
        if isinstance(images, list):
            x = torch.stack(images, dim=0)
        else:
            x = images

        feature_map = self.features(x)
        outputs = []
        for batch_index in range(x.shape[0]):
            response = feature_map[batch_index, 0]
            score = torch.sigmoid(response[16:48, 16:48].mean() * 10.0)
            outputs.append(
                {
                    "boxes": torch.tensor([[16.0, 16.0, 48.0, 48.0]], device=x.device),
                    "labels": torch.tensor([1], device=x.device),
                    "scores": score.unsqueeze(0),
                }
            )
        return outputs


@pytest.fixture(scope="module")
def toy_image() -> np.ndarray:
    """Create a synthetic image with a bright square in the target bbox."""
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:48, 16:48] = 255
    return image


@pytest.fixture(scope="module")
def toy_detection() -> Detection:
    """Create a synthetic detection aligned with the toy model's response area."""
    return Detection(
        bbox=[16, 16, 48, 48],
        class_id=0,
        class_name="person",
        confidence=0.95,
        area=(48 - 16) * (48 - 16),
        relative_size="medium",
        detection_id=0,
    )


@pytest.fixture(scope="module")
def toy_model() -> ToyDetectionModel:
    """Instantiate the toy detector model once for the test module."""
    model = ToyDetectionModel().eval()
    return model


@pytest.fixture(scope="module")
def toy_target_layer(toy_model: ToyDetectionModel):
    """Return the target convolutional layer for CAM-based tests."""
    return toy_model.features[0]


@pytest.fixture(scope="module")
def fasterrcnn_detector() -> DetectorWrapper:
    """Load Faster R-CNN once for the GradCAM integration test."""
    pytest.importorskip("torchvision")
    detector = DetectorWrapper("fasterrcnn_resnet50_fpn_v2")
    detector.load_model()
    return detector


@pytest.mark.slow
def test_output_shape_gradcam(sample_image: Path, fasterrcnn_detector: DetectorWrapper) -> None:
    """GradCAM saliency map has the same H, W as the input image."""
    pytest.importorskip("captum")

    image = load_image(str(sample_image))
    detections = fasterrcnn_detector.detect(image)
    if not detections:
        pytest.skip("Detector did not produce any detections on the sample image.")

    explainer = GradCAMExplainer(_load_method_config("gradcam"))
    result = explainer.explain(
        fasterrcnn_detector.get_model(),
        image,
        detections[0],
        fasterrcnn_detector.get_target_layer(),
    )
    assert result.map.shape == (image.shape[0], image.shape[1])


def test_values_normalized(toy_image: np.ndarray, toy_detection: Detection, toy_model: ToyDetectionModel, toy_target_layer) -> None:
    """All saliency values remain in the range [0, 1]."""
    pytest.importorskip("captum")
    pytest.importorskip("skimage")
    pytest.importorskip("sklearn")

    explainers = [
        GradCAMExplainer(_load_method_config("gradcam")),
        GCAMEExplainer(_load_method_config("gcame")),
        DCLOSEExplainer({**_load_method_config("dclose"), "num_masks_dev": 30, "segmentation_scales": [20, 40]}),
        LIMEExplainer({**_load_method_config("lime"), "num_perturbations": 40, "num_superpixels": 20}),
    ]

    for explainer in explainers:
        result = explainer.explain(
            toy_model,
            toy_image,
            toy_detection,
            toy_target_layer if isinstance(explainer, (GradCAMExplainer, GCAMEExplainer)) else None,
        )
        assert result.map.min() >= 0.0
        assert result.map.max() <= 1.0


def test_saliency_nonzero_in_bbox(
    toy_image: np.ndarray,
    toy_detection: Detection,
    toy_model: ToyDetectionModel,
    toy_target_layer,
) -> None:
    """Saliency inside the bbox should exceed the image-wide mean."""
    pytest.importorskip("captum")

    result = GradCAMExplainer(_load_method_config("gradcam")).explain(
        toy_model,
        toy_image,
        toy_detection,
        toy_target_layer,
    )

    x1, y1, x2, y2 = toy_detection.bbox
    bbox_mean = float(result.map[y1:y2, x1:x2].mean())
    total_mean = float(result.map.mean())
    assert bbox_mean > total_mean


def test_computation_time_logged(
    toy_image: np.ndarray,
    toy_detection: Detection,
    toy_model: ToyDetectionModel,
    toy_target_layer,
) -> None:
    """Each SaliencyMap records a positive floating-point computation time."""
    pytest.importorskip("captum")

    result = GradCAMExplainer(_load_method_config("gradcam")).explain(
        toy_model,
        toy_image,
        toy_detection,
        toy_target_layer,
    )
    assert result.computation_time > 0
    assert isinstance(result.computation_time, float)


def test_gradcam_faster_than_dclose(
    toy_image: np.ndarray,
    toy_detection: Detection,
    toy_model: ToyDetectionModel,
    toy_target_layer,
) -> None:
    """GradCAM should be at least an order of magnitude faster than D-CLOSE."""
    pytest.importorskip("captum")
    pytest.importorskip("skimage")

    gradcam_result = GradCAMExplainer(_load_method_config("gradcam")).explain(
        toy_model,
        toy_image,
        toy_detection,
        toy_target_layer,
    )
    dclose_result = DCLOSEExplainer(
        {**_load_method_config("dclose"), "num_masks_dev": 300, "segmentation_scales": [20, 40, 60]}
    ).explain(
        toy_model,
        toy_image,
        toy_detection,
        None,
    )
    assert gradcam_result.computation_time * 10 < dclose_result.computation_time


def test_factory_function() -> None:
    """get_explainer returns the correct explainer class for each method."""
    assert isinstance(get_explainer("gradcam", {}), GradCAMExplainer)
    assert isinstance(get_explainer("gcame", {}), GCAMEExplainer)
    assert isinstance(get_explainer("dclose", {}), DCLOSEExplainer)
    assert isinstance(get_explainer("lime", {}), LIMEExplainer)


def test_factory_unknown_method() -> None:
    """get_explainer raises ValueError for an unknown method name."""
    with pytest.raises(ValueError):
        get_explainer("unknown_method", {})


def test_normalize_edge_cases() -> None:
    """normalize_saliency handles all-zero and constant arrays."""
    assert np.allclose(XAIExplainer.normalize_saliency(np.zeros((10, 10))), 0)
    assert np.allclose(XAIExplainer.normalize_saliency(np.ones((10, 10)) * 5), 0)
