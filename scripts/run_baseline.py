"""CLI scaffold for reproducing ODExAI-style object detection explanation baselines."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BaselineRunConfig:
    """Configuration bundle for the baseline reproduction entry point."""

    detector_config: str = "config/detector_config.yaml"
    xai_config: str = "config/xai_config.yaml"
    eval_config: str = "config/eval_config.yaml"
    output_dir: str = "results/exp1_xai_selection"
    image_limit: int = 500


class BaselineRunner:
    """Runner responsible for baseline reproduction and result export."""

    def __init__(self, config: BaselineRunConfig) -> None:
        """Initialize the baseline runner from parsed CLI settings."""
        raise NotImplementedError("TODO: Stage 2")

    def run(self) -> None:
        """Execute the configured baseline experiment suite."""
        raise NotImplementedError("TODO: Stage 2")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the baseline script."""
    raise NotImplementedError("TODO: Stage 2")


def main() -> None:
    """Parse arguments and launch baseline reproduction."""
    raise NotImplementedError("TODO: Stage 2")


if __name__ == "__main__":
    main()

