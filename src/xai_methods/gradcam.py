"""Grad-CAM explainer for object detection models."""

from __future__ import annotations

import gc
import logging
import time
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from .base import (
    SaliencyMap,
    XAIExplainer,
    _extract_target_score_from_output,
    _finalize_saliency_map,
    _forward_detector_from_tensor,
    _prepare_model_input,
    _require_torch,
)

if TYPE_CHECKING:
    from ..detector import Detection

logger = logging.getLogger(__name__)


class _DetectionTargetModule:
    """Captum-compatible wrapper that exposes one detection score per image."""

    def __new__(cls, base_model: Any, detection: "Detection", meta: Mapping[str, Any]) -> Any:
        torch = _require_torch()

        class _Wrapped(torch.nn.Module):
            def __init__(self, base_model: Any, detection: "Detection", meta: Mapping[str, Any]) -> None:
                super().__init__()
                self.base_model = base_model
                self.detection = detection
                self.meta = dict(meta)

            def forward(self, x: Any) -> Any:
                if x.ndim == 3:
                    x = x.unsqueeze(0)

                scores = []
                for batch_index in range(x.shape[0]):
                    single_x = x[batch_index : batch_index + 1]
                    output = _forward_detector_from_tensor(
                        self.base_model,
                        single_x,
                        str(self.meta.get("family", "generic")),
                    )
                    score = _extract_target_score_from_output(
                        output=output,
                        detection=self.detection,
                        model=self.base_model,
                        meta=self.meta,
                        fallback_tensor=single_x,
                    )
                    scores.append(score)

                return torch.stack(scores, dim=0).unsqueeze(1)

        return _Wrapped(base_model, detection, meta)


class GradCAMExplainer(XAIExplainer):
    """Standard Grad-CAM explanation for object detection."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        defaults = {"target_layers": {}, "upsample_mode": "bilinear"}
        defaults.update(dict(config))
        super().__init__(defaults)

    def explain(
        self,
        model: Any,
        image: np.ndarray,
        detection: "Detection",
        target_layer: Any | None = None,
    ) -> SaliencyMap:
        """Generate a Grad-CAM saliency map for the requested detection."""
        if target_layer is None:
            raise ValueError("GradCAMExplainer requires a target_layer.")

        t_start = time.time()
        torch = _require_torch()
        try:
            from captum.attr import LayerGradCam
        except ImportError as exc:
            raise ImportError("captum is required for GradCAMExplainer.") from exc

        input_tensor, meta = _prepare_model_input(model, image)
        wrapper = _DetectionTargetModule(model, detection, meta)
        layer_gradcam = LayerGradCam(wrapper, target_layer)

        try:
            if hasattr(model, "zero_grad"):
                model.zero_grad(set_to_none=True)
            attribution = layer_gradcam.attribute(input_tensor, target=0)
            cam = attribution.detach().cpu().numpy()
            cam = np.squeeze(cam)
            if cam.ndim == 3:
                cam = cam.mean(axis=0)
            cam = np.maximum(cam, 0.0)
            cam = _finalize_saliency_map(cam, image.shape[:2], detection)
        except Exception as exc:
            logger.warning(
                "GradCAM failed for detection %s; falling back to bbox prior. Error: %s",
                detection.detection_id,
                exc,
            )
            cam = _finalize_saliency_map(np.zeros(image.shape[:2], dtype=np.float32), image.shape[:2], detection)
        finally:
            del layer_gradcam
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return SaliencyMap(
            map=cam,
            method_name="gradcam",
            computation_time=float(time.time() - t_start),
            detection_id=detection.detection_id,
        )
