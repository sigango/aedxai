"""CLI scaffold for running the AED-XAI ablation experiment suite."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ExperimentSuiteConfig:
    """Configuration bundle for the five AED-XAI ablation experiments."""

    detector_config: str = "config/detector_config.yaml"
    vlm_config: str = "config/vlm_config.yaml"
    xai_config: str = "config/xai_config.yaml"
    eval_config: str = "config/eval_config.yaml"
    experiments: list[str] = field(
        default_factory=lambda: [
            "exp1_xai_selection",
            "exp2_feedback_loop",
            "exp3_vlm_ablation",
            "exp4_detector_generalization",
            "exp5_cross_domain",
        ]
    )
    results_root: str = "results"


class ExperimentRunner:
    """Runner responsible for orchestrating all configured ablation studies."""

    def __init__(self, config: ExperimentSuiteConfig) -> None:
        """Initialize the ablation runner from parsed settings."""
        raise NotImplementedError("TODO: Stage 2")

    def run(self) -> None:
        """Execute the selected experiment suite and persist outputs."""
        raise NotImplementedError("TODO: Stage 2")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the experiment suite."""
    raise NotImplementedError("TODO: Stage 2")


def main() -> None:
    """Parse arguments and launch the ablation suite."""
    raise NotImplementedError("TODO: Stage 2")


if __name__ == "__main__":
    main()

