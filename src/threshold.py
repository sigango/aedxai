"""Adaptive threshold computation for the AED-XAI feedback loop.

Supports three modes:
  - "fixed"      : constant tau read from config (backward-compatible default)
  - "percentile" : tau = percentile_p of the current batch composite scores
  - "learned"    : tau predicted by a tiny linear model from batch statistics
                   trained on a held-out validation set

Paper framing:
  The percentile mode is the primary ablation; the learned mode is presented
  as a stronger baseline in Table 4 (Threshold Ablation).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class AdaptiveThreshold:
    """Compute an adaptive refinement threshold from a batch of composite scores.

    Args:
        mode: One of "fixed", "percentile", or "learned".
        fixed_value: Tau used when mode="fixed" (default 0.5).
        percentile: Integer percentile p used when mode="percentile" (default 40).
            A value of 40 means "refine the bottom 40% of detections".
        model_path: Path to a persisted learned-threshold model (pickle).
            Required only when mode="learned" and you want to load a
            pre-trained predictor; ignored otherwise.
    """

    SUPPORTED_MODES = {"fixed", "percentile", "learned"}

    def __init__(
        self,
        mode: str = "percentile",
        fixed_value: float = 0.5,
        percentile: int = 40,
        model_path: str | None = None,
    ) -> None:
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unknown threshold mode '{mode}'. "
                f"Choose from {sorted(self.SUPPORTED_MODES)}."
            )
        self.mode = mode
        self.fixed_value = float(fixed_value)
        self.percentile = int(np.clip(percentile, 1, 99))
        self._learned_model: Any = None

        if model_path is not None and Path(model_path).exists():
            self._load_learned_model(model_path)

    def compute(self, scores: list[float] | np.ndarray) -> float:
        """Return the threshold tau appropriate for this batch of scores."""
        arr = np.asarray(scores, dtype=np.float64)

        if arr.size == 0:
            logger.debug(
                "AdaptiveThreshold.compute: empty scores, using fixed_value=%.3f",
                self.fixed_value,
            )
            return self.fixed_value

        if self.mode == "fixed":
            return self.fixed_value

        if self.mode == "percentile":
            tau = float(np.percentile(arr, self.percentile))
            tau = float(np.clip(tau, 0.0, 1.0))
            logger.debug(
                "AdaptiveThreshold (percentile=%d): tau=%.4f  [min=%.4f mean=%.4f max=%.4f]",
                self.percentile,
                tau,
                arr.min(),
                arr.mean(),
                arr.max(),
            )
            return tau

        if self.mode == "learned":
            return self._predict_learned(arr)

        return self.fixed_value

    def fit(
        self,
        val_batch_stats: list[dict[str, float]],
        val_optimal_thresholds: list[float],
        save_path: str | None = None,
    ) -> None:
        """Train the learned threshold predictor on a validation set."""
        try:
            from sklearn.linear_model import Ridge
        except ImportError as exc:
            raise ImportError("scikit-learn is required for learned threshold mode.") from exc

        X = np.array(
            [[s["mean"], s["std"], s["median"]] for s in val_batch_stats],
            dtype=np.float64,
        )
        y = np.array(val_optimal_thresholds, dtype=np.float64)
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        self._learned_model = model
        logger.info(
            "Learned threshold model trained on %d validation batches. "
            "Coefficients: mean=%.4f std=%.4f median=%.4f intercept=%.4f",
            len(val_batch_stats),
            model.coef_[0],
            model.coef_[1],
            model.coef_[2],
            model.intercept_,
        )

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with Path(save_path).open("wb") as handle:
                pickle.dump(model, handle)
            logger.info("Saved learned threshold model to %s", save_path)

    def _predict_learned(self, arr: np.ndarray) -> float:
        if self._learned_model is None:
            logger.warning(
                "Learned threshold mode selected but no model loaded. "
                "Falling back to percentile mode (p=%d).",
                self.percentile,
            )
            tau = float(np.percentile(arr, self.percentile))
            return float(np.clip(tau, 0.0, 1.0))

        stats = np.array([[arr.mean(), arr.std(), np.median(arr)]])
        tau = float(self._learned_model.predict(stats)[0])
        tau = float(np.clip(tau, 0.0, 1.0))
        logger.debug("AdaptiveThreshold (learned): tau=%.4f", tau)
        return tau

    def _load_learned_model(self, model_path: str) -> None:
        with Path(model_path).open("rb") as handle:
            self._learned_model = pickle.load(handle)
        logger.info("Loaded learned threshold model from %s", model_path)

    @classmethod
    def from_config(cls, feedback_config: dict) -> "AdaptiveThreshold":
        """Construct from the feedback subsection of eval_config.yaml."""
        mode = str(feedback_config.get("threshold_mode", "percentile"))
        fixed_value = float(feedback_config.get("threshold", 0.5))
        percentile = int(feedback_config.get("threshold_percentile", 40))
        model_path = feedback_config.get("threshold_model_path", None)
        return cls(
            mode=mode,
            fixed_value=fixed_value,
            percentile=percentile,
            model_path=model_path if model_path else None,
        )


__all__ = ["AdaptiveThreshold"]
