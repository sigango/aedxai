"""D-CLOSE explainer for black-box object detection models."""

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
class DCloseConfig:
    """Configuration bundle for D-CLOSE explanation generation."""

    num_masks_dev: int = 200
    num_masks_final: int = 2000
    segmentation_method: str = "felzenszwalb"
    segmentation_scales: list[int] | None = None
    batch_size: int = 32
    similarity_metric: str = "confidence_drop"


class DCLOSEExplainer(XAIExplainer):
    """Detection CLOSEst-neighbor Explanation using region perturbations."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Initialize D-CLOSE with merged method defaults."""
        defaults = {
            "num_masks_dev": 200,
            "num_masks_final": 2000,
            "segmentation_method": "felzenszwalb",
            "segmentation_scales": [50, 100, 200],
            "batch_size": 32,
            "similarity_metric": "confidence_drop",
        }
        defaults.update(dict(config))
        super().__init__(defaults)

    def segment_image(self, image: np.ndarray) -> np.ndarray:
        """Segment an image using the primary configured segmentation method."""
        try:
            from skimage.segmentation import felzenszwalb, slic
        except ImportError as exc:
            raise ImportError("scikit-image is required for DCLOSEExplainer.") from exc

        if str(self.config.get("segmentation_method", "felzenszwalb")).lower() == "slic":
            return slic(
                image,
                n_segments=int(self.config.get("segmentation_scales", [100])[0]),
                compactness=10,
                start_label=0,
                channel_axis=-1,
            )
        return felzenszwalb(image, scale=int(self.config.get("segmentation_scales", [100])[0]), sigma=0.5, min_size=50)

    def generate_masks(self, segments: np.ndarray, final_mode: bool = False) -> np.ndarray:
        """Generate segmentation-based binary masks."""
        rng = np.random.default_rng(0)
        num_masks = int(self.config["num_masks_final"] if final_mode else self.config["num_masks_dev"])
        n_segments = int(segments.max()) + 1
        masks = []

        for _ in range(max(1, num_masks)):
            keep_count = max(1, n_segments // 2)
            chosen = rng.choice(n_segments, size=keep_count, replace=False)
            mask = np.isin(segments, chosen).astype(np.float32)
            masks.append(mask)

        return np.stack(masks, axis=0).astype(np.float32)

    def explain(
        self,
        model: Any,
        image: np.ndarray,
        detection: "Detection",
        target_layer: Any | None = None,
    ) -> SaliencyMap:
        """Generate a D-CLOSE saliency map for one detection."""
        del target_layer
        t_start = time.time()
        image = np.ascontiguousarray(image.astype(np.uint8))

        try:
            from skimage.segmentation import felzenszwalb, slic
        except ImportError as exc:
            raise ImportError("scikit-image is required for DCLOSEExplainer.") from exc

        segmentation_method = str(self.config.get("segmentation_method", "felzenszwalb")).lower()
        segmentation_scales = list(self.config.get("segmentation_scales", [50, 100, 200]))
        num_masks = int(self.config.get("num_masks_dev", 200))
        per_scale_masks = max(1, num_masks // max(1, len(segmentation_scales)))
        batch_size = max(1, int(self.config.get("batch_size", 32)))
        rng = np.random.default_rng(0)

        all_masks: list[np.ndarray] = []
        for scale in segmentation_scales:
            if segmentation_method == "slic":
                segments = slic(
                    image,
                    n_segments=max(8, int(scale)),
                    compactness=10,
                    start_label=0,
                    channel_axis=-1,
                )
            else:
                segments = felzenszwalb(image, scale=int(scale), sigma=0.5, min_size=50)

            n_segments = int(segments.max()) + 1
            for _ in range(per_scale_masks):
                keep_count = max(1, n_segments // 2)
                chosen = rng.choice(n_segments, size=keep_count, replace=False)
                all_masks.append(np.isin(segments, chosen).astype(np.float32))

        if not all_masks:
            all_masks = [np.ones(image.shape[:2], dtype=np.float32)]

        scores: list[float] = []
        for batch_start in range(0, len(all_masks), batch_size):
            mask_batch = all_masks[batch_start : batch_start + batch_size]
            for mask in mask_batch:
                masked_image = (image.astype(np.float32) * mask[:, :, None]).astype(np.uint8)
                output, _, meta = _forward_detector(model, masked_image, require_grad=False)
                matched_confidence, matched_iou = _extract_matching_confidence_from_output(
                    output=output,
                    detection=detection,
                    model=model,
                    meta=meta,
                )
                confidence_ratio = matched_confidence / max(float(detection.confidence), 1e-6)
                similarity = matched_iou * confidence_ratio
                scores.append(float(similarity))

        masks_array = np.stack(all_masks, axis=0).astype(np.float32)
        scores_array = np.asarray(scores, dtype=np.float32)
        saliency = (scores_array[:, None, None] * masks_array).sum(axis=0)
        density = masks_array.sum(axis=0) + 1e-8
        saliency = saliency / density
        saliency = _finalize_saliency_map(saliency, image.shape[:2], detection)

        gc.collect()
        return SaliencyMap(
            map=saliency,
            method_name="dclose",
            computation_time=float(time.time() - t_start),
            detection_id=detection.detection_id,
        )


DCloseExplainer = DCLOSEExplainer
