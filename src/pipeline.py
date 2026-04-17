"""End-to-end pipeline orchestration for the AED-XAI research system."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import yaml

from . import utils
from .detector import DetectorWrapper
from .evaluator import AutoEvaluator
from .feedback_loop import FeedbackLoop
from .vlm_judge import VLMJudge
from .xai_methods import get_explainer
from .xai_selector import XAISelector

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
        self.detector_config_path = str(detector_config_path)
        self.vlm_config_path = str(vlm_config_path)
        self.xai_config_path = str(xai_config_path)
        self.eval_config_path = str(eval_config_path)

        self.detector: DetectorWrapper | None = None
        self.vlm_judge: VLMJudge | None = None
        self.xai_selector: XAISelector | None = None
        self.evaluator: AutoEvaluator | None = None
        self.feedback_loop: FeedbackLoop | None = None
        self.xai_methods: dict[str, Any] = {}

        self._xai_config = self._load_yaml(self.xai_config_path)
        xai_section = dict(self._xai_config.get("xai", {}))
        methods_section = dict(xai_section.get("methods", {}))
        self.enabled_xai_methods = [
            method_name
            for method_name, method_config in methods_section.items()
            if bool(dict(method_config).get("enabled", False))
        ]

        logger.info(
            "Initialized AEDXAIPipeline with enabled XAI methods: %s",
            ", ".join(self.enabled_xai_methods) if self.enabled_xai_methods else "none",
        )

    def setup(self) -> None:
        """Load configuration files and initialize all pipeline components."""
        if self.detector is None:
            self.detector = DetectorWrapper(config_path=self.detector_config_path)
            self.detector.load_model()

        if self.vlm_judge is None:
            self.vlm_judge = VLMJudge(config_path=self.vlm_config_path)

        if self.evaluator is None:
            self.evaluator = AutoEvaluator(config_path=self.eval_config_path)

        if self.feedback_loop is None:
            self.feedback_loop = FeedbackLoop(config_path=self.eval_config_path)

        if self.xai_selector is None:
            selector_checkpoint = self._resolve_path("data/checkpoints/xai_selector.pth")
            if selector_checkpoint.exists():
                self.xai_selector = XAISelector(model_path=str(selector_checkpoint))
                logger.info("Loaded XAI selector checkpoint from %s", selector_checkpoint)
            else:
                self.xai_selector = XAISelector()
                logger.info("No XAI selector checkpoint found; using rule-based fallback.")

        if not self.xai_methods:
            methods_config = dict(self._xai_config.get("xai", {}).get("methods", {}))
            for method_name, method_config in methods_config.items():
                method_options = dict(method_config)
                if not bool(method_options.get("enabled", False)):
                    continue
                self.xai_methods[method_name] = get_explainer(method_name, method_options)

        if torch.cuda.is_available():
            vram_gb = torch.cuda.memory_allocated() / 1e9
            logger.info("AED-XAI setup complete. CUDA memory allocated: %.2f GB", vram_gb)
        else:
            logger.info("AED-XAI setup complete on CPU.")

    def run_on_image(self, image_path: str) -> PipelineResult:
        """Process one image through the full AED-XAI pipeline."""
        self.setup()
        assert self.detector is not None
        assert self.vlm_judge is not None
        assert self.xai_selector is not None
        assert self.evaluator is not None
        assert self.feedback_loop is not None

        try:
            image = utils.load_image(image_path)
            self.vlm_judge.load_model()

            feedback_result = self.feedback_loop.run(
                image=image,
                image_path=image_path,
                detector=self.detector,
                vlm_judge=self.vlm_judge,
                xai_selector=self.xai_selector,
                xai_methods=self.xai_methods,
                evaluator=self.evaluator,
            )

            final_eval_results = feedback_result.final_eval_results
            mean_composite = (
                float(np.mean([result.composite for result in final_eval_results]))
                if final_eval_results
                else None
            )

            return PipelineResult(
                image_path=image_path,
                detections=feedback_result.final_detections,
                assessments=[],
                saliency_maps=feedback_result.final_saliency_maps,
                evaluation_results=final_eval_results,
                composite_score=mean_composite,
                metadata={
                    "iterations": len(feedback_result.iterations),
                    "converged": feedback_result.converged,
                },
            )
        except Exception:
            logger.exception("AED-XAI pipeline failed for image %s", image_path)
            raise
        finally:
            if self.vlm_judge is not None:
                self.vlm_judge.unload_model()

    def run_batch(self, image_paths: list[str]) -> list[PipelineResult]:
        """Process a batch of images through the full AED-XAI pipeline."""
        results: list[PipelineResult] = []
        total = len(image_paths)

        for index, image_path in enumerate(image_paths, start=1):
            logger.info("Running AED-XAI pipeline on image %d/%d: %s", index, total, image_path)
            try:
                results.append(self.run_on_image(image_path))
            except Exception as exc:
                logger.exception("Failed to process image %s", image_path)
                results.append(
                    PipelineResult(
                        image_path=image_path,
                        detections=[],
                        assessments=[],
                        saliency_maps=[],
                        evaluation_results=[],
                        composite_score=None,
                        metadata={"error": str(exc)},
                    )
                )

        return results

    def shutdown(self) -> None:
        """Release detector, VLM, and auxiliary resources."""
        if self.detector is not None:
            self.detector.unload_model()
        if self.vlm_judge is not None:
            self.vlm_judge.unload_model()

        self.detector = None
        self.vlm_judge = None
        self.xai_selector = None
        self.evaluator = None
        self.feedback_loop = None
        self.xai_methods = {}

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _resolve_path(path: str) -> Path:
        """Resolve a possibly relative path against the project root."""
        raw_path = Path(path)
        if raw_path.is_absolute():
            return raw_path

        candidates = [
            Path.cwd() / raw_path,
            Path(__file__).resolve().parents[1] / raw_path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _load_yaml(self, path: str) -> dict[str, Any]:
        """Load a YAML file into a dictionary."""
        resolved = self._resolve_path(path)
        with resolved.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
