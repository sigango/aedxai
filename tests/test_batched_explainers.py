"""Batched LIME / D-CLOSE forwards must give the same maps as one-at-a-time forwards."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from src.xai_methods import get_explainer
from toy_models import ToyDetectionModel, make_detection, toy_image


def _grid_segments(image, n_segments=16, **_kwargs):
    height, width = image.shape[:2]
    side = max(1, int(round(np.sqrt(n_segments))))
    rows = np.minimum((np.arange(height) * side) // height, side - 1)
    cols = np.minimum((np.arange(width) * side) // width, side - 1)
    return (rows[:, None] * side + cols[None, :]).astype(np.int64)


@pytest.fixture()
def fake_skimage(monkeypatch):
    """Grid segmentation stand-in so the test does not need scikit-image."""
    segmentation = types.ModuleType("skimage.segmentation")
    segmentation.slic = lambda image, n_segments=16, **kw: _grid_segments(image, n_segments)
    segmentation.felzenszwalb = lambda image, scale=100, **kw: _grid_segments(image, max(4, 400 // int(scale)))
    package = types.ModuleType("skimage")
    package.segmentation = segmentation
    monkeypatch.setitem(sys.modules, "skimage", package)
    monkeypatch.setitem(sys.modules, "skimage.segmentation", segmentation)


@pytest.mark.parametrize(
    "method, config",
    [
        ("lime", {"num_perturbations": 60, "num_superpixels": 16}),
        ("dclose", {"num_masks_dev": 30, "segmentation_scales": [50, 100]}),
    ],
)
def test_batched_forward_matches_single(fake_skimage, method, config) -> None:
    model = ToyDetectionModel().eval()
    image, detection = toy_image(), make_detection([16, 16, 48, 48])
    single = get_explainer(method, {**config, "batch_size": 1}).explain(model, image, detection)
    batched = get_explainer(method, {**config, "batch_size": 16}).explain(model, image, detection)
    assert np.allclose(single.map, batched.map, atol=1e-5)
    assert single.is_fallback == batched.is_fallback
