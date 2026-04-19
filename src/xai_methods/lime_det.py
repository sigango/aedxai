"""LIME adapted for object detection outputs."""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from .base import (
    SaliencyMap,
    XAIExplainer,
    _extract_matching_confidence_from_output,
    _finalize_saliency_map,
    _forward_detector,
)

if TYPE_CHECKING:
    from ..detector import Detection

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LimeDetectionConfig:
    """Configuration bundle for LIME explanation generation."""

    num_superpixels: int = 50
    num_perturbations: int = 500
    segmentation_method: str = "slic"
    slic_compactness: int = 10
    kernel_width: float = 0.25
    batch_size: int = 32


class LIMEExplainer(XAIExplainer):
    """LIME adapted from classification to object detection confidence."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Initialize LIME with merged method defaults."""
        defaults = {
            "num_superpixels": 50,
            "num_perturbations": 500,
            "segmentation_method": "slic",
            "slic_compactness": 10,
            "kernel_width": 0.25,
            "batch_size": 32,
        }
        defaults.update(dict(config))
        super().__init__(defaults)

    def segment_image(self, image: np.ndarray) -> np.ndarray:
        """Segment an image into superpixels for local perturbation analysis."""
        try:
            from skimage.segmentation import slic
        except ImportError as exc:
            raise ImportError("scikit-image is required for LIMEExplainer.") from exc

        return slic(
            image,
            n_segments=int(self.config.get("num_superpixels", 50)),
            compactness=float(self.config.get("slic_compactness", 10)),
            start_label=0,
            channel_axis=-1,
        )

    def explain(
        self,
        model: Any,
        image: np.ndarray,
        detection: "Detection",
        target_layer: Any | None = None,
    ) -> SaliencyMap:
        """Generate a LIME saliency map for one detection."""
        del target_layer
        t_start = time.time()
        image = np.ascontiguousarray(image.astype(np.uint8))
        segments = self.segment_image(image)
        n_superpixels = int(segments.max()) + 1

        num_perturbations = int(self.config.get("num_perturbations", 500))
        batch_size = max(1, int(self.config.get("batch_size", 32)))
        kernel_width = max(float(self.config.get("kernel_width", 0.25)), 1e-6)
        rng = np.random.default_rng(0)

        perturbation_matrix = rng.binomial(1, 0.5, size=(num_perturbations, n_superpixels)).astype(np.int32)
        perturbation_matrix[0] = 1

        predictions: list[float] = []
        # Fill value 114 matches YOLOX letterbox padding (and is a neutral gray for FRCNN),
        # keeping masked regions in-distribution. Black (0) used previously is out-of-distribution
        # for YOLOX and biases the detector's confidence drop.
        fill_value = 114
        for batch_start in range(0, num_perturbations, batch_size):
            batch = perturbation_matrix[batch_start : batch_start + batch_size]
            for row in batch:
                selected_superpixels = np.where(row == 1)[0]
                mask = np.isin(segments, selected_superpixels)
                masked_image = image.copy()
                masked_image[~mask] = fill_value

                output, _, meta = _forward_detector(model, masked_image, require_grad=False)
                matched_confidence, _ = _extract_matching_confidence_from_output(
                    output=output,
                    detection=detection,
                    model=model,
                    meta=meta,
                )
                predictions.append(float(matched_confidence))

        try:
            from sklearn.linear_model import Ridge
        except ImportError as exc:
            raise ImportError("scikit-learn is required for LIMEExplainer.") from exc

        # Cosine-like distance: fraction of superpixels turned off, normalized to [0, 1]
        # Standard LIME uses distance from the reference (all-ones) perturbation.
        # Normalizing by n_superpixels keeps kernel_width scale-independent.
        n_superpixels_f = float(max(n_superpixels, 1))
        distances = np.sqrt((1 - perturbation_matrix).sum(axis=1).astype(np.float64) / n_superpixels_f).astype(
            np.float32
        )
        kernel = np.exp(-(distances**2) / (kernel_width**2)).astype(np.float32)

        ridge = Ridge(alpha=1.0)
        ridge.fit(perturbation_matrix, np.asarray(predictions, dtype=np.float32), sample_weight=kernel)
        importance = np.asarray(ridge.coef_, dtype=np.float32)

        saliency = np.zeros(image.shape[:2], dtype=np.float32)
        for superpixel_id in range(n_superpixels):
            saliency[segments == superpixel_id] = max(float(importance[superpixel_id]), 0.0)

        saliency = _finalize_saliency_map(saliency, image.shape[:2], detection)
        gc.collect()
        return SaliencyMap(
            map=saliency,
            method_name="lime",
            computation_time=float(time.time() - t_start),
            detection_id=detection.detection_id,
        )


LimeDetectionExplainer = LIMEExplainer
