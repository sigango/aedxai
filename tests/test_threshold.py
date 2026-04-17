"""Unit tests for src/threshold.py AdaptiveThreshold."""

from __future__ import annotations

import numpy as np
import pytest

from src.threshold import AdaptiveThreshold


def test_fixed_mode() -> None:
    t = AdaptiveThreshold(mode="fixed", fixed_value=0.5)
    assert t.compute([0.1, 0.2, 0.9]) == pytest.approx(0.5)


def test_fixed_mode_empty() -> None:
    t = AdaptiveThreshold(mode="fixed", fixed_value=0.4)
    assert t.compute([]) == pytest.approx(0.4)


def test_percentile_mode_basic() -> None:
    scores = [0.2, 0.4, 0.6, 0.8, 1.0]
    t = AdaptiveThreshold(mode="percentile", percentile=40)
    tau = t.compute(scores)
    expected = float(np.percentile(scores, 40))
    assert tau == pytest.approx(expected, abs=1e-6)


def test_percentile_mode_clips_to_unit() -> None:
    t = AdaptiveThreshold(mode="percentile", percentile=99)
    tau = t.compute([1.5, 2.0])
    assert 0.0 <= tau <= 1.0


def test_percentile_empty_fallback() -> None:
    t = AdaptiveThreshold(mode="percentile", fixed_value=0.3, percentile=40)
    assert t.compute([]) == pytest.approx(0.3)


def test_learned_mode_falls_back_without_model() -> None:
    t = AdaptiveThreshold(mode="learned", percentile=40)
    scores = [0.1, 0.3, 0.5, 0.7]
    tau = t.compute(scores)
    expected = float(np.percentile(scores, 40))
    assert tau == pytest.approx(expected, abs=1e-6)


def test_learned_mode_fit_and_predict() -> None:
    pytest.importorskip("sklearn")
    t = AdaptiveThreshold(mode="learned", percentile=40)
    val_stats = [
        {"mean": 0.4, "std": 0.1, "median": 0.38},
        {"mean": 0.6, "std": 0.15, "median": 0.62},
        {"mean": 0.3, "std": 0.05, "median": 0.29},
    ]
    optimal_thresholds = [0.35, 0.55, 0.28]
    t.fit(val_stats, optimal_thresholds)
    tau = t.compute([0.3, 0.4, 0.5])
    assert 0.0 <= tau <= 1.0


def test_from_config_percentile() -> None:
    cfg = {"threshold": 0.5, "threshold_mode": "percentile", "threshold_percentile": 35}
    t = AdaptiveThreshold.from_config(cfg)
    assert t.mode == "percentile"
    assert t.percentile == 35


def test_from_config_fixed() -> None:
    cfg = {"threshold": 0.6, "threshold_mode": "fixed"}
    t = AdaptiveThreshold.from_config(cfg)
    assert t.mode == "fixed"
    assert t.compute([0.1, 0.9]) == pytest.approx(0.6)


def test_invalid_mode() -> None:
    with pytest.raises(ValueError, match="Unknown threshold mode"):
        AdaptiveThreshold(mode="magic")
