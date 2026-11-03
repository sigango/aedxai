"""Factory helpers for AED-XAI explanation method implementations."""

from __future__ import annotations

import re
from typing import Any, Mapping

from .base import SaliencyMap, XAIExplainer
from .controls import BBoxPriorExplainer, CenterPriorExplainer, RandomExplainer, shuffle_saliency
from .dclose import DCLOSEExplainer, DCloseExplainer
from .gcame import GCAMEExplainer
from .gradcam import GradCAMExplainer
from .lime_det import LIMEExplainer, LimeDetectionExplainer

EXPLAINER_REGISTRY = {
    "gradcam": GradCAMExplainer,
    "gcame": GCAMEExplainer,
    "dclose": DCLOSEExplainer,
    "lime": LIMEExplainer,
    "bbox_prior": BBoxPriorExplainer,
    "center_prior": CenterPriorExplainer,
    "random": RandomExplainer,
}

# Variants that remove the detector-box conditioning built into a base method,
# so the audit can measure how much of a score comes from that conditioning.
# name -> (base registry key, config overrides applied on top of the caller's config)
EXPLAINER_VARIANTS: dict[str, tuple[str, dict[str, Any]]] = {
    "gcame_raw": ("gcame", {"use_bbox_center_weighting": False}),
    "gradcam_raw": ("gradcam", {"renormalize_in_bbox": False, "zero_outside_bbox": False}),
}

# Methods that need the detector's CAM target layer.
CAM_METHODS = frozenset({"gradcam", "gcame", "gradcam_raw", "gcame_raw"})

# Model-free controls. ``bbox_prior_sNNN`` is the bbox prior with sigma scaled by
# NNN/100 (``bbox_prior`` = ``bbox_prior_s100``), used by the matched-prior audit.
CONTROL_METHODS = frozenset({"bbox_prior", "center_prior", "random"})
_SCALED_PRIOR = re.compile(r"^bbox_prior_s(\d{3})$")


def scaled_prior_name(sigma_scale: float) -> str:
    """Control name of a bbox prior whose sigma is ``sigma_scale`` x half the box side."""
    return f"bbox_prior_s{int(round(sigma_scale * 100)):03d}"


def is_control_method(method_name: str) -> bool:
    """True for model-free controls, including scaled bbox priors."""
    name = method_name.strip().lower()
    return name in CONTROL_METHODS or _SCALED_PRIOR.match(name) is not None


def get_explainer(method_name: str, config: Mapping[str, Any]) -> XAIExplainer:
    """Factory function to create an XAI explainer by registry or variant name."""
    normalized_name = method_name.strip().lower()
    if normalized_name in EXPLAINER_VARIANTS:
        base_name, overrides = EXPLAINER_VARIANTS[normalized_name]
        merged = {**dict(config), **overrides}
        return EXPLAINER_REGISTRY[base_name](merged)
    scaled = _SCALED_PRIOR.match(normalized_name)
    if scaled is not None:
        return BBoxPriorExplainer({**dict(config), "sigma_scale": int(scaled.group(1)) / 100.0, "name": normalized_name})
    if normalized_name not in EXPLAINER_REGISTRY:
        raise ValueError(
            f"Unknown XAI method: {method_name}. "
            f"Available: {list(EXPLAINER_REGISTRY.keys()) + list(EXPLAINER_VARIANTS.keys())}"
        )
    return EXPLAINER_REGISTRY[normalized_name](dict(config))


def base_method_name(method_name: str) -> str:
    """Return the registry key whose config section a (possibly variant) method uses."""
    normalized_name = method_name.strip().lower()
    if normalized_name in EXPLAINER_VARIANTS:
        return EXPLAINER_VARIANTS[normalized_name][0]
    if _SCALED_PRIOR.match(normalized_name):
        return "bbox_prior"
    return normalized_name


__all__ = [
    "BBoxPriorExplainer",
    "CAM_METHODS",
    "CONTROL_METHODS",
    "CenterPriorExplainer",
    "DCLOSEExplainer",
    "DCloseExplainer",
    "EXPLAINER_REGISTRY",
    "EXPLAINER_VARIANTS",
    "GCAMEExplainer",
    "GradCAMExplainer",
    "LIMEExplainer",
    "LimeDetectionExplainer",
    "RandomExplainer",
    "SaliencyMap",
    "XAIExplainer",
    "base_method_name",
    "get_explainer",
    "is_control_method",
    "scaled_prior_name",
    "shuffle_saliency",
]
