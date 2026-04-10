"""Tests for AED-XAI annotation-free explanation evaluation metrics."""

from __future__ import annotations

import numpy as np

from src.evaluator import AutoEvaluator


def test_pg_perfect() -> None:
    """Saliency peak inside bbox should yield a perfect pointing-game score."""
    saliency = np.zeros((100, 100), dtype=np.float32)
    saliency[50, 50] = 1.0
    bbox = [25, 25, 75, 75]
    evaluator = AutoEvaluator()
    assert evaluator.pointing_game(saliency, bbox) == 1.0


def test_pg_miss() -> None:
    """Saliency peak outside bbox should yield a miss."""
    saliency = np.zeros((100, 100), dtype=np.float32)
    saliency[5, 5] = 1.0
    bbox = [25, 25, 75, 75]
    evaluator = AutoEvaluator()
    assert evaluator.pointing_game(saliency, bbox) == 0.0


def test_ebpg_concentrated() -> None:
    """Energy concentrated inside the bbox should produce a high EBPG score."""
    saliency = np.zeros((100, 100), dtype=np.float32)
    saliency[30:70, 30:70] = 1.0
    bbox = [25, 25, 75, 75]
    evaluator = AutoEvaluator()
    assert evaluator.energy_based_pg(saliency, bbox) > 0.9


def test_ebpg_scattered() -> None:
    """Mostly external saliency energy should produce a low EBPG score."""
    saliency = np.ones((100, 100), dtype=np.float32)
    bbox = [45, 45, 55, 55]
    evaluator = AutoEvaluator()
    assert evaluator.energy_based_pg(saliency, bbox) < 0.1


def test_sparsity_focused() -> None:
    """A one-hot saliency map should have very high sparsity."""
    saliency = np.zeros((100, 100), dtype=np.float32)
    saliency[50, 50] = 1.0
    evaluator = AutoEvaluator()
    assert evaluator.sparsity_gini(saliency) > 0.9


def test_sparsity_uniform() -> None:
    """A uniform saliency map should have low sparsity."""
    saliency = np.ones((100, 100), dtype=np.float32) * 0.5
    evaluator = AutoEvaluator()
    assert evaluator.sparsity_gini(saliency) < 0.1


def test_composite_in_range() -> None:
    """Composite scores should remain in the stable [0, 1] range."""
    evaluator = AutoEvaluator()
    for pg in [0.0, 0.5, 1.0]:
        for oa in [-0.5, 0.0, 0.5, 1.0]:
            for sparsity in [0.0, 0.5, 1.0]:
                score = evaluator.composite_score({"pg": pg, "oa": oa, "sparsity": sparsity})
                assert 0.0 <= score <= 1.0
