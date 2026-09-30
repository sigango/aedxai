"""Model-free control explanations used to audit annotation-free XAI metrics.

These maps use no model evidence beyond the detector's own box (bbox prior) or
none at all (center prior, random). Scoring them with the same evaluator shows
how much of a metric can be earned by box geometry alone.
"""

from __future__ import annotations

import time
import zlib
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from .base import SaliencyMap, XAIExplainer, _bbox_prior_map

if TYPE_CHECKING:
    from ..detector import Detection


class BBoxPriorExplainer(XAIExplainer):
    """Gaussian centred on the detection box (sigma = half the box side)."""

    def explain(
        self,
        model: Any,
        image: np.ndarray,
        detection: "Detection",
        target_layer: Any | None = None,
    ) -> SaliencyMap:
        """Return the bbox-prior map; model and target layer are ignored."""
        del model, target_layer
        t_start = time.time()
        saliency = _bbox_prior_map(image.shape[:2], detection)
        return SaliencyMap(
            map=saliency,
            method_name="bbox_prior",
            computation_time=float(time.time() - t_start),
            detection_id=detection.detection_id,
        )


class CenterPriorExplainer(XAIExplainer):
    """Gaussian centred on the image, independent of the detection."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        defaults = {"sigma_fraction": 0.25}
        defaults.update(dict(config))
        super().__init__(defaults)

    def explain(
        self,
        model: Any,
        image: np.ndarray,
        detection: "Detection",
        target_layer: Any | None = None,
    ) -> SaliencyMap:
        """Return an image-centred Gaussian; the detection only supplies the id."""
        del model, target_layer
        t_start = time.time()
        height, width = image.shape[:2]
        sigma_fraction = float(self.config.get("sigma_fraction", 0.25))
        sigma_y = max(1.0, sigma_fraction * height)
        sigma_x = max(1.0, sigma_fraction * width)
        grid_y, grid_x = np.mgrid[0:height, 0:width]
        prior = np.exp(
            -(
                ((grid_x - width / 2.0) ** 2) / (2.0 * sigma_x**2)
                + ((grid_y - height / 2.0) ** 2) / (2.0 * sigma_y**2)
            )
        )
        return SaliencyMap(
            map=XAIExplainer.normalize_saliency(prior),
            method_name="center_prior",
            computation_time=float(time.time() - t_start),
            detection_id=detection.detection_id,
        )


class RandomExplainer(XAIExplainer):
    """Uniform random map, seeded per detection so reruns are identical."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        defaults = {"seed": 0}
        defaults.update(dict(config))
        super().__init__(defaults)

    def explain(
        self,
        model: Any,
        image: np.ndarray,
        detection: "Detection",
        target_layer: Any | None = None,
    ) -> SaliencyMap:
        """Return a random map seeded by (config seed, detection box)."""
        del model, target_layer
        t_start = time.time()
        rng = np.random.default_rng(_detection_seed(int(self.config.get("seed", 0)), detection))
        saliency = rng.random(image.shape[:2], dtype=np.float32)
        return SaliencyMap(
            map=XAIExplainer.normalize_saliency(saliency),
            method_name="random",
            computation_time=float(time.time() - t_start),
            detection_id=detection.detection_id,
        )


def shuffle_saliency(saliency: np.ndarray, seed: int = 0) -> np.ndarray:
    """Permute pixels of a map: keeps its value histogram (and Gini), destroys layout."""
    array = np.asarray(saliency, dtype=np.float32)
    rng = np.random.default_rng(seed)
    flat = array.reshape(-1).copy()
    rng.shuffle(flat)
    return flat.reshape(array.shape)


def _detection_seed(base_seed: int, detection: "Detection") -> int:
    """Derive a stable per-detection seed from the box coordinates and class."""
    key = f"{base_seed}:{detection.class_id}:{':'.join(str(int(v)) for v in detection.bbox)}"
    return zlib.crc32(key.encode("utf-8"))


__all__ = [
    "BBoxPriorExplainer",
    "CenterPriorExplainer",
    "RandomExplainer",
    "shuffle_saliency",
]
