"""Factory helpers for AED-XAI explanation method implementations."""

from __future__ import annotations

from typing import Any, Mapping

from .base import SaliencyMap, XAIExplainer
from .dclose import DCLOSEExplainer, DCloseExplainer
from .gcame import GCAMEExplainer
from .gradcam import GradCAMExplainer
from .lime_det import LIMEExplainer, LimeDetectionExplainer

EXPLAINER_REGISTRY = {
    "gradcam": GradCAMExplainer,
    "gcame": GCAMEExplainer,
    "dclose": DCLOSEExplainer,
    "lime": LIMEExplainer,
}


def get_explainer(method_name: str, config: Mapping[str, Any]) -> XAIExplainer:
    """Factory function to create an XAI explainer by registry name."""
    normalized_name = method_name.strip().lower()
    if normalized_name not in EXPLAINER_REGISTRY:
        raise ValueError(
            f"Unknown XAI method: {method_name}. "
            f"Available: {list(EXPLAINER_REGISTRY.keys())}"
        )
    return EXPLAINER_REGISTRY[normalized_name](dict(config))


__all__ = [
    "DCLOSEExplainer",
    "DCloseExplainer",
    "EXPLAINER_REGISTRY",
    "GCAMEExplainer",
    "GradCAMExplainer",
    "LIMEExplainer",
    "LimeDetectionExplainer",
    "SaliencyMap",
    "XAIExplainer",
    "get_explainer",
]
