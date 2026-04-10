"""Shared pytest fixtures for AED-XAI smoke tests and development."""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pytest

from src.detector import Detection

try:
    from src.xai_methods.base import SaliencyMap
except ImportError:
    SaliencyMap = None

logger = logging.getLogger(__name__)

COCO_SAMPLE_IMAGE_URL = "http://images.cocodataset.org/val2017/000000000139.jpg"


@pytest.fixture(scope="session")
def sample_image() -> Path:
    """Download a stable COCO sample image into the local test fixture folder."""
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    image_path = fixtures_dir / "000000000139.jpg"
    if not image_path.exists():
        logger.info("Downloading sample COCO fixture to %s", image_path)
        urlretrieve(COCO_SAMPLE_IMAGE_URL, image_path)

    return image_path


@pytest.fixture()
def sample_detections() -> list[Detection]:
    """Return three deterministic mock detections with known bounding boxes."""
    return [
        Detection(
            bbox=[64, 48, 192, 320],
            class_id=0,
            class_name="person",
            confidence=0.94,
            area=(192 - 64) * (320 - 48),
            relative_size="large",
            detection_id=0,
        ),
        Detection(
            bbox=[220, 80, 408, 352],
            class_id=2,
            class_name="car",
            confidence=0.87,
            area=(408 - 220) * (352 - 80),
            relative_size="large",
            detection_id=1,
        ),
        Detection(
            bbox=[420, 96, 576, 288],
            class_id=16,
            class_name="dog",
            confidence=0.79,
            area=(576 - 420) * (288 - 96),
            relative_size="large",
            detection_id=2,
        ),
    ]


@pytest.fixture()
def mock_saliency_map(sample_detections: list[Detection]):
    """Return a synthetic Gaussian saliency map centered on the first bbox."""
    if SaliencyMap is None:
        pytest.skip("SaliencyMap not available yet (requires XAI base module).")

    detection = sample_detections[0]
    height, width = 384, 640

    x1, y1, x2, y2 = detection.bbox
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    sigma = max((x2 - x1), (y2 - y1)) / 4.0

    grid_y, grid_x = np.mgrid[0:height, 0:width]
    gaussian = np.exp(-(((grid_x - center_x) ** 2) + ((grid_y - center_y) ** 2)) / (2.0 * sigma**2))
    gaussian = gaussian.astype(np.float32)
    gaussian /= np.max(gaussian)

    return SaliencyMap(
        map=gaussian,
        method_name="gradcam",
        computation_time=0.012,
        detection_id=detection.detection_id,
    )
