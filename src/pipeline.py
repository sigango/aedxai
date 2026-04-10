"""End-to-end pipeline orchestration for the AED-XAI research system."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .detector import Detection
    from .evaluator import EvaluationResult
    from .vlm_judge import VLMAssessment
    from .xai_methods.base import SaliencyMap

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineConfigPaths:
    """Filesystem locations for the pipeline's YAML configuration files."""

    detector_config_path: str
    vlm_config_path: str
    xai_config_path: str
    eval_config_path: str


@dataclass(slots=True)
class PipelineResult:
    """Aggregated outputs produced for one processed image."""

    image_path: str
    detections: list["Detection"] = field(default_factory=list)
    assessments: list["VLMAssessment"] = field(default_factory=list)
    saliency_maps: list["SaliencyMap"] = field(default_factory=list)
    evaluation_results: list["EvaluationResult"] = field(default_factory=list)
    composite_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AEDXAIPipeline:
    """High-level coordinator for detection, judging, explanation, and feedback."""

    def __init__(
        self,
        detector_config_path: str,
        vlm_config_path: str,
        xai_config_path: str,
        eval_config_path: str,
    ) -> None:
        """Initialize the pipeline from its configuration file paths."""
        raise NotImplementedError("TODO: Stage 2")

    def setup(self) -> None:
        """Load configuration files and initialize all pipeline components."""
        raise NotImplementedError("TODO: Stage 2")

    def run_on_image(self, image_path: str) -> PipelineResult:
        """Process one image through the full AED-XAI pipeline."""
        raise NotImplementedError("TODO: Stage 2")

    def run_batch(self, image_paths: list[str]) -> list[PipelineResult]:
        """Process a batch of images through the full AED-XAI pipeline."""
        raise NotImplementedError("TODO: Stage 2")

    def shutdown(self) -> None:
        """Release detector, VLM, and auxiliary resources."""
        raise NotImplementedError("TODO: Stage 2")

