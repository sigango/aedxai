"""G-CAME explainer for object detection models."""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import cv2
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
    import torch

    from ..detector import Detection

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GCAMEConfig:
    """Configuration bundle for GCAME explanation generation."""

    target_layers: dict[str, str]
    gaussian_sigma: float = 2.0
    gaussian_kernel_size: int = 11
    use_bbox_center_weighting: bool = True


class GCAMEExplainer(XAIExplainer):
    """Gradient-weighted CAM with Gaussian enhancement."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Initialize GCAME with merged method defaults."""
        defaults = {
            "target_layers": {},
            "gaussian_sigma": 2.0,
            "gaussian_kernel_size": 11,
            "use_bbox_center_weighting": True,
        }
        defaults.update(dict(config))
        super().__init__(defaults)

    def explain(
        self,
        model: Any,
        image: np.ndarray,
        detection: "Detection",
        target_layer: Any | None = None,
    ) -> SaliencyMap:
        """Generate a G-CAME saliency map for one detection."""
        if target_layer is None:
            raise ValueError("GCAMEExplainer requires a target_layer.")

        t_start = time.time()
        torch = _require_torch()
        input_tensor, meta = _prepare_model_input(model, image)

        activations: dict[str, Any] = {}
        gradients: dict[str, Any] = {}

        def _forward_hook(_module: Any, _inputs: Any, output: Any) -> None:
            activations["value"] = output

        def _backward_hook(_module: Any, _grad_input: Any, grad_output: Any) -> None:
            if grad_output:
                gradients["value"] = grad_output[0]

        handle_fwd = target_layer.register_forward_hook(_forward_hook)
        handle_bwd = target_layer.register_full_backward_hook(_backward_hook)

        try:
            if hasattr(model, "zero_grad"):
                model.zero_grad(set_to_none=True)

            with torch.enable_grad():
                output = _forward_detector_from_tensor(model, input_tensor, str(meta.get("family", "generic")))
                target_score = _extract_target_score_from_output(
                    output=output,
                    detection=detection,
                    model=model,
                    meta=meta,
                    fallback_tensor=input_tensor,
                )
                target_score.backward()

            acts = activations["value"]
            grads = gradients["value"]
            if acts.ndim == 4:
                acts = acts[0]
            if grads.ndim == 4:
                grads = grads[0]

            weights = grads.mean(dim=(1, 2))
            cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))
            cam_np = cam.detach().cpu().numpy().astype(np.float32)
            cam_np = cv2.resize(cam_np, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

            if bool(self.config.get("use_bbox_center_weighting", True)):
                x1, y1, x2, y2 = [int(value) for value in detection.bbox]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                box_scale = max(1.0, max(x2 - x1, y2 - y1))
                sigma = max(1.0, float(self.config.get("gaussian_sigma", 2.0)) * box_scale)
                grid_y, grid_x = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
                gaussian = np.exp(-(((grid_x - cx) ** 2) + ((grid_y - cy) ** 2)) / (2.0 * sigma**2)).astype(
                    np.float32
                )
                cam_np = cam_np * gaussian

            cam_np = _finalize_saliency_map(cam_np, image.shape[:2], detection)
        except Exception as exc:
            logger.warning(
                "GCAME failed for detection %s; falling back to bbox prior. Error: %s",
                detection.detection_id,
                exc,
            )
            cam_np = _finalize_saliency_map(np.zeros(image.shape[:2], dtype=np.float32), image.shape[:2], detection)
        finally:
            handle_fwd.remove()
            handle_bwd.remove()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return SaliencyMap(
            map=cam_np,
            method_name="gcame",
            computation_time=float(time.time() - t_start),
            detection_id=detection.detection_id,
        )
